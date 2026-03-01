#!/usr/bin/env python3
"""
Deploy Branham LLM API to Google Cloud Run from local Docker build.

Flow (mirrors work deploy script pattern):
  1. Read .env → build Cloud Run --set-env-vars string
  2. Ensure Artifact Registry repo exists
  3. docker build --platform linux/amd64
  4. docker tag + docker push to Artifact Registry
  5. gcloud run deploy with env vars, resource limits, scaling

Prerequisites:
  - gcloud CLI installed and authenticated:
      gcloud auth login admin@branhamsermons.ai
      gcloud config set project <PROJECT_ID>
  - Docker Desktop running
  - .env file with API keys

Usage:
  uv run python scripts/deploy_cloudrun.py
  uv run python scripts/deploy_cloudrun.py --dry-run   # preview only
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ───────────────────────────────────────────────────
PROJECT_ID = "elevated-codex-487017-a6"
REGION = "us-central1"
SERVICE_NAME = "branham-llm-api"
AR_REPO = "branham-llm-api"
AR_DOMAIN = f"{REGION}-docker.pkg.dev"
IMAGE_NAME = f"{AR_DOMAIN}/{PROJECT_ID}/{AR_REPO}/{SERVICE_NAME}"

# Cloud Run resource settings (from WORKING_PROGRESS.md targets)
MEMORY = "4Gi"
CPU = "2"
MAX_INSTANCES = 5
MIN_INSTANCES = 1
CONCURRENCY = 10
REQUEST_TIMEOUT = 300
PORT = 8080
# Startup budget: model warm ≈ 30s + uvicorn boot ≈ 5s + buffer
STARTUP_CPU_BOOST = True


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


def ensure_apis() -> None:
    """Enable required GCP APIs if not already enabled."""
    apis = [
        "run.googleapis.com",
        "artifactregistry.googleapis.com",
    ]
    for api in apis:
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


def docker_build(image_tag: str) -> None:
    """Build Docker image locally for linux/amd64."""
    run([
        "docker", "build",
        "--platform", "linux/amd64",
        "-t", image_tag,
        ".",
    ])


def docker_push(image_tag: str) -> None:
    """Authenticate Docker and push to Artifact Registry."""
    run(["gcloud", "auth", "configure-docker", AR_DOMAIN, "--quiet"])
    run(["docker", "push", image_tag])


def cloud_run_deploy(image_tag: str, env_vars: dict[str, str]) -> str:
    """Deploy (or update) Cloud Run service. Returns the service URL."""
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

    result = run(cmd, capture=True)
    service_url = (result.stdout or "").strip()
    return service_url


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    print("=" * 70)
    print("Branham LLM API — Cloud Run Deployment")
    print("=" * 70)
    print(f"Project:    {PROJECT_ID}")
    print(f"Region:     {REGION}")
    print(f"Service:    {SERVICE_NAME}")
    print(f"Memory:     {MEMORY}")
    print(f"CPU:        {CPU}")
    print(f"Scaling:    {MIN_INSTANCES}–{MAX_INSTANCES} instances")
    print(f"Concurrency:{CONCURRENCY}")
    print("=" * 70)

    # 0) Auth + APIs
    ensure_gcloud_auth()
    ensure_apis()

    # 1) Load .env
    env_vars = load_env_vars(project_root / ".env")
    print(f"\nLoaded {len(env_vars)} env vars from .env")
    for k in sorted(env_vars.keys()):
        print(f"  {k} = {env_vars[k][:8]}...")

    # 3) Build image tag
    timestamp = int(time.time())
    image_tag = f"{IMAGE_NAME}:v{timestamp}"
    print(f"\nImage tag: {image_tag}")

    if dry_run:
        print("\n[DRY RUN] Would build, push, and deploy. Exiting.")
        return

    # 2) Ensure Artifact Registry (only when actually deploying)
    ensure_artifact_registry()

    print("\n" + "=" * 70)
    print("STEP 1: Building Docker image (this may take several minutes)")
    print("=" * 70)
    docker_build(image_tag)

    print("\n" + "=" * 70)
    print("STEP 2: Pushing image to Artifact Registry")
    print("=" * 70)
    docker_push(image_tag)

    print("\n" + "=" * 70)
    print("STEP 3: Deploying to Cloud Run")
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
