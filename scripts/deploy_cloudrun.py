#!/usr/bin/env python3
"""
Deploy Branham LLM API to Google Cloud Run.

Supported build modes:
  - local: Docker builds/pushes from the local machine
  - cloudbuild: Google Cloud Build builds and pushes remotely

Flow:
  1. Read .env → build Cloud Run --set-env-vars string
  2. Ensure required APIs are enabled
  3. Ensure Artifact Registry repo exists
  4. Build/push image via selected build mode
  5. gcloud run deploy with env vars, resource limits, scaling

Prerequisites:
  - gcloud CLI installed and authenticated:
      gcloud auth login admin@branhamsermons.ai
      gcloud config set project <PROJECT_ID>
  - local mode only: Docker Desktop running
  - .env file with API keys

Usage:
  uv run python scripts/deploy_cloudrun.py
  uv run python scripts/deploy_cloudrun.py --build-mode local
  uv run python scripts/deploy_cloudrun.py --build-mode cloudbuild
  uv run python scripts/deploy_cloudrun.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

# ── Configuration ───────────────────────────────────────────────────
PROJECT_ID = "elevated-codex-487017-a6"
REGION = "us-central1"
SERVICE_NAME = "branham-llm-api"
AR_REPO = "branham-llm-api"
AR_DOMAIN = f"{REGION}-docker.pkg.dev"
IMAGE_NAME = f"{AR_DOMAIN}/{PROJECT_ID}/{AR_REPO}/{SERVICE_NAME}"

# Cloud Run resource settings.
# Startup footprint is ~3.2 GB (Qwen3-Embedding + FAISS flat-IP + BM25 + chunk
# store), which is tight on 4Gi under burst load. 6Gi gives genuine headroom
# without doubling steady-state cost. Concurrency 6 spreads heavy moments
# across instances rather than piling onto one.
MEMORY = "6Gi"
CPU = "2"
MAX_INSTANCES = 5
MIN_INSTANCES = 1
CONCURRENCY = 6
REQUEST_TIMEOUT = 300
PORT = 8080
# Startup budget: model warm ≈ 30s + uvicorn boot ≈ 5s + buffer
STARTUP_CPU_BOOST = True
BUILD_MODES = ("local", "cloudbuild")
REQUIRED_RUNTIME_FILES = (
    "Dockerfile",
    "docker/entrypoint.sh",
    "config/default.yaml",
    "scripts/prefetch_hf_model.py",
    "data/indices/bm25.index",
    "data/indices/bm25_doc_map.jsonl",
    "data/indices/bm25_meta.json",
    "data/indices/bm25_vocab.json",
    "data/indices/faiss.index",
    "data/indices/faiss_id_map.jsonl",
    "data/indices/faiss_meta.json",
    "data/processed/chunks.sqlite",
    "data/reference/biography.txt",
)


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command with logging."""
    print(f"\n  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def load_env_vars(env_path: Path) -> dict[str, str]:
    """Parse .env file into a dict (skips comments and empty lines)."""
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        print(f"WARNING: {env_path} not found — no env vars will be injected")
        return env_vars
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            env_vars[key] = value
    return env_vars


def validate_required_runtime_files(project_root: Path) -> None:
    """Fail fast when deployment-critical runtime artifacts are missing."""
    missing = [
        rel_path for rel_path in REQUIRED_RUNTIME_FILES
        if not (project_root / rel_path).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {rel_path}" for rel_path in missing)
        raise FileNotFoundError(
            "Missing runtime deployment artifacts required by Dockerfile/Cloud Run:\n"
            f"{formatted}"
        )


def list_cloudbuild_upload_files() -> set[str]:
    """Return the file paths that gcloud will upload for Cloud Build."""
    result = run(
        ["gcloud", "meta", "list-files-for-upload"],
        capture=True,
    )
    return {
        line.strip()
        for line in (result.stdout or "").splitlines()
        if line.strip()
    }


def validate_cloudbuild_upload_files(project_root: Path) -> None:
    """
    Fail fast when .gcloudignore excludes runtime artifacts required by Dockerfile.

    Local file existence is not enough for Cloud Build; the files must also be part
    of the uploaded source context.
    """
    _ = project_root  # cwd is already set to project_root in main()
    upload_files = list_cloudbuild_upload_files()
    missing = [
        rel_path for rel_path in REQUIRED_RUNTIME_FILES
        if rel_path not in upload_files
    ]
    if missing:
        formatted = "\n".join(f"  - {rel_path}" for rel_path in missing)
        raise FileNotFoundError(
            "Cloud Build source upload is missing runtime artifacts required by Dockerfile.\n"
            "These files exist locally but are excluded from the uploaded build context.\n"
            "Check .gcloudignore re-include rules.\n"
            f"{formatted}"
        )


def ensure_gcloud_auth() -> None:
    """Verify gcloud is authenticated and project is set."""
    result = run(
        ["gcloud", "config", "get-value", "project"],
        capture=True, check=False,
    )
    current_project = (result.stdout or "").strip()
    if current_project != PROJECT_ID:
        print(f"\nSetting gcloud project to {PROJECT_ID}")
        run(["gcloud", "config", "set", "project", PROJECT_ID])

    result = run(
        ["gcloud", "config", "get-value", "account"],
        capture=True, check=False,
    )
    account = (result.stdout or "").strip()
    if not account:
        print("\nERROR: No gcloud account active. Run:")
        print("  gcloud auth login admin@branhamsermons.ai")
        sys.exit(1)
    print(f"\ngcloud account: {account}")
    print(f"gcloud project: {PROJECT_ID}")


def required_apis(build_mode: str) -> list[str]:
    """Return the GCP APIs needed for the selected build mode."""
    apis = [
        "run.googleapis.com",
        "artifactregistry.googleapis.com",
    ]
    if build_mode == "cloudbuild":
        apis.append("cloudbuild.googleapis.com")
    return apis


def ensure_apis(build_mode: str) -> None:
    """Enable required GCP APIs if not already enabled."""
    for api in required_apis(build_mode):
        result = run(
            ["gcloud", "services", "list", "--enabled",
             "--filter", f"name:{api}", "--format", "value(name)"],
            capture=True, check=False,
        )
        if api not in (result.stdout or ""):
            print(f"Enabling API: {api}")
            run(["gcloud", "services", "enable", api, "--project", PROJECT_ID])
        else:
            print(f"API already enabled: {api}")


def ensure_artifact_registry() -> None:
    """Create Artifact Registry Docker repo if it doesn't exist."""
    result = run(
        ["gcloud", "artifacts", "repositories", "describe", AR_REPO,
         "--location", REGION, "--format", "value(name)"],
        capture=True, check=False,
    )
    if result.returncode != 0:
        print(f"\nCreating Artifact Registry repo: {AR_REPO}")
        run([
            "gcloud", "artifacts", "repositories", "create", AR_REPO,
            "--repository-format", "docker",
            "--location", REGION,
            "--description", "Branham LLM API Docker images",
        ])
    else:
        print(f"Artifact Registry repo exists: {AR_REPO}")


def build_local_docker_cmd(image_tag: str) -> list[str]:
    """Return the local Docker build command."""
    return [
        "docker", "build",
        "--platform", "linux/amd64",
        "-t", image_tag,
        ".",
    ]


def docker_build(image_tag: str) -> None:
    """Build Docker image locally for linux/amd64."""
    run(build_local_docker_cmd(image_tag))


def build_docker_push_cmd(image_tag: str) -> list[str]:
    """Return the local Docker push command."""
    return ["docker", "push", image_tag]


def docker_push(image_tag: str) -> None:
    """Authenticate Docker and push to Artifact Registry."""
    run(["gcloud", "auth", "configure-docker", AR_DOMAIN, "--quiet"])
    run(build_docker_push_cmd(image_tag))


def build_cloud_build_submit_cmd(image_tag: str) -> list[str]:
    """Return the Cloud Build submit command."""
    return [
        "gcloud", "builds", "submit",
        "--project", PROJECT_ID,
        "--tag", image_tag,
        ".",
    ]


def cloud_build_submit(image_tag: str) -> None:
    """Build and push the image remotely via Cloud Build."""
    run(build_cloud_build_submit_cmd(image_tag))


def build_cloud_run_deploy_cmd(image_tag: str, env_vars: dict[str, str]) -> list[str]:
    """Return the Cloud Run deploy command."""
    env_string = ",".join(f"{k}={v}" for k, v in env_vars.items())
    cmd = [
        "gcloud", "run", "deploy", SERVICE_NAME,
        "--image", image_tag,
        "--region", REGION,
        "--platform", "managed",
        "--port", str(PORT),
        "--memory", MEMORY,
        "--cpu", str(CPU),
        "--max-instances", str(MAX_INSTANCES),
        "--min-instances", str(MIN_INSTANCES),
        "--concurrency", str(CONCURRENCY),
        "--timeout", str(REQUEST_TIMEOUT),
        "--allow-unauthenticated",
        "--no-cpu-throttling",
        "--format", "value(status.url)",
    ]
    if STARTUP_CPU_BOOST:
        cmd.append("--cpu-boost")
    if env_string:
        cmd.extend(["--set-env-vars", env_string])
    return cmd


def cloud_run_deploy(image_tag: str, env_vars: dict[str, str]) -> str:
    """Deploy (or update) Cloud Run service. Returns the service URL."""
    result = run(build_cloud_run_deploy_cmd(image_tag, env_vars), capture=True)
    service_url = (result.stdout or "").strip()
    return service_url


def build_image_tag(*, timestamp: int | None = None) -> str:
    """Return a versioned Artifact Registry image tag."""
    tag_ts = timestamp if timestamp is not None else int(time.time())
    return f"{IMAGE_NAME}:v{tag_ts}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for deployment mode selection."""
    parser = argparse.ArgumentParser(description="Deploy Branham LLM API to Cloud Run.")
    parser.add_argument(
        "--build-mode",
        choices=BUILD_MODES,
        default="local",
        help="How to build/push the container image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected deployment plan without executing it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dry_run = args.dry_run
    build_mode = args.build_mode

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    print("=" * 70)
    print("Branham LLM API — Cloud Run Deployment")
    print("=" * 70)
    print(f"Project:    {PROJECT_ID}")
    print(f"Region:     {REGION}")
    print(f"Service:    {SERVICE_NAME}")
    print(f"Build mode: {build_mode}")
    print(f"Memory:     {MEMORY}")
    print(f"CPU:        {CPU}")
    print(f"Scaling:    {MIN_INSTANCES}–{MAX_INSTANCES} instances")
    print(f"Concurrency:{CONCURRENCY}")
    print("=" * 70)

    # 0) Auth + APIs
    ensure_gcloud_auth()
    ensure_apis(build_mode)

    # 1) Load .env
    env_vars = load_env_vars(project_root / ".env")
    print(f"\nLoaded {len(env_vars)} env vars from .env")
    for k in sorted(env_vars.keys()):
        print(f"  {k} = {env_vars[k][:8]}...")

    # 2) Validate deployment-critical files before any build starts.
    validate_required_runtime_files(project_root)
    if build_mode == "cloudbuild":
        validate_cloudbuild_upload_files(project_root)

    # 3) Build image tag
    image_tag = build_image_tag()
    print(f"\nImage tag: {image_tag}")

    if dry_run:
        if build_mode == "local":
            print("\n[DRY RUN] Would run local Docker build/push, then deploy.")
        else:
            print("\n[DRY RUN] Would run Cloud Build submit, then deploy.")
        return

    # 2) Ensure Artifact Registry (only when actually deploying)
    ensure_artifact_registry()

    print("\n" + "=" * 70)
    if build_mode == "local":
        print("STEP 1: Building Docker image locally (this may take several minutes)")
    else:
        print("STEP 1: Building image via Cloud Build (this may take several minutes)")
    print("=" * 70)
    if build_mode == "local":
        docker_build(image_tag)

        print("\n" + "=" * 70)
        print("STEP 2: Pushing image to Artifact Registry")
        print("=" * 70)
        docker_push(image_tag)
    else:
        cloud_build_submit(image_tag)

    print("\n" + "=" * 70)
    if build_mode == "local":
        print("STEP 3: Deploying to Cloud Run")
    else:
        print("STEP 2: Deploying to Cloud Run")
    print("=" * 70)
    service_url = cloud_run_deploy(image_tag, env_vars)

    print("\n" + "=" * 70)
    print("DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"Service URL: {service_url}")
    print(f"Image:       {image_tag}")
    print(f"Health:      {service_url}/api/health")
    print(f"Chat:        POST {service_url}/api/chat")
    print("=" * 70)


if __name__ == "__main__":
    main()
