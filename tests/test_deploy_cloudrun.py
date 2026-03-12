from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pytest


def _load_module():
    root = Path(__file__).resolve().parent.parent
    script_path = root / "scripts" / "deploy_cloudrun.py"
    spec = importlib.util.spec_from_file_location("deploy_cloudrun", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_cloudrun_test_module"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_local_mode():
    module = _load_module()
    args = module.parse_args([])
    assert args.build_mode == "local"
    assert args.dry_run is False


def test_parse_args_accepts_cloudbuild_mode():
    module = _load_module()
    args = module.parse_args(["--build-mode", "cloudbuild", "--dry-run"])
    assert args.build_mode == "cloudbuild"
    assert args.dry_run is True


def test_required_apis_include_cloudbuild_only_for_cloudbuild_mode():
    module = _load_module()
    assert "cloudbuild.googleapis.com" not in module.required_apis("local")
    assert "cloudbuild.googleapis.com" in module.required_apis("cloudbuild")


def test_build_cloud_build_submit_cmd_matches_expected_gcloud_shape():
    module = _load_module()
    image_tag = "us-central1-docker.pkg.dev/proj/repo/svc:v123"
    assert module.build_cloud_build_submit_cmd(image_tag) == [
        "gcloud",
        "builds",
        "submit",
        "--project",
        module.PROJECT_ID,
        "--tag",
        image_tag,
        ".",
    ]


def test_build_cloud_run_deploy_cmd_includes_env_vars_and_cpu_boost():
    module = _load_module()
    cmd = module.build_cloud_run_deploy_cmd("img:v1", {"A": "1", "B": "two"})
    assert cmd[:4] == ["gcloud", "run", "deploy", module.SERVICE_NAME]
    assert "--image" in cmd and "img:v1" in cmd
    assert "--region" in cmd and module.REGION in cmd
    assert "--cpu-boost" in cmd
    env_idx = cmd.index("--set-env-vars")
    assert cmd[env_idx + 1] == "A=1,B=two"


def test_validate_required_runtime_files_raises_for_missing_artifacts(tmp_path):
    module = _load_module()
    with pytest.raises(FileNotFoundError) as exc_info:
        module.validate_required_runtime_files(tmp_path)
    assert "Missing runtime deployment artifacts" in str(exc_info.value)
    assert "data/indices/bm25.index" in str(exc_info.value)


def test_main_routes_local_mode_without_cloudbuild(monkeypatch):
    module = _load_module()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "ensure_gcloud_auth", lambda: calls.append(("auth", None)))
    monkeypatch.setattr(module, "ensure_apis", lambda mode: calls.append(("apis", mode)))
    monkeypatch.setattr(module, "load_env_vars", lambda path: {"X": "1"})
    monkeypatch.setattr(module, "validate_required_runtime_files", lambda project_root: calls.append(("validate", project_root)))
    monkeypatch.setattr(module, "ensure_artifact_registry", lambda: calls.append(("repo", None)))
    monkeypatch.setattr(module, "docker_build", lambda tag: calls.append(("docker_build", tag)))
    monkeypatch.setattr(module, "docker_push", lambda tag: calls.append(("docker_push", tag)))
    monkeypatch.setattr(module, "cloud_build_submit", lambda tag: calls.append(("cloud_build", tag)))
    monkeypatch.setattr(module, "cloud_run_deploy", lambda tag, env: calls.append(("deploy", (tag, env))) or "https://svc")
    monkeypatch.setattr(module.time, "time", lambda: 1234567890)
    monkeypatch.setattr(module.os, "chdir", lambda path: None)

    module.main(["--build-mode", "local"])

    assert ("apis", "local") in calls
    assert any(name == "validate" for name, _ in calls)
    assert any(name == "docker_build" for name, _ in calls)
    assert any(name == "docker_push" for name, _ in calls)
    assert not any(name == "cloud_build" for name, _ in calls)
    assert any(name == "deploy" for name, _ in calls)


def test_main_routes_cloudbuild_mode_without_local_docker(monkeypatch):
    module = _load_module()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(module, "ensure_gcloud_auth", lambda: calls.append(("auth", None)))
    monkeypatch.setattr(module, "ensure_apis", lambda mode: calls.append(("apis", mode)))
    monkeypatch.setattr(module, "load_env_vars", lambda path: {"X": "1"})
    monkeypatch.setattr(module, "validate_required_runtime_files", lambda project_root: calls.append(("validate", project_root)))
    monkeypatch.setattr(module, "ensure_artifact_registry", lambda: calls.append(("repo", None)))
    monkeypatch.setattr(module, "docker_build", lambda tag: calls.append(("docker_build", tag)))
    monkeypatch.setattr(module, "docker_push", lambda tag: calls.append(("docker_push", tag)))
    monkeypatch.setattr(module, "cloud_build_submit", lambda tag: calls.append(("cloud_build", tag)))
    monkeypatch.setattr(module, "cloud_run_deploy", lambda tag, env: calls.append(("deploy", (tag, env))) or "https://svc")
    monkeypatch.setattr(module.time, "time", lambda: 1234567890)
    monkeypatch.setattr(module.os, "chdir", lambda path: None)

    module.main(["--build-mode", "cloudbuild"])

    assert ("apis", "cloudbuild") in calls
    assert any(name == "validate" for name, _ in calls)
    assert any(name == "cloud_build" for name, _ in calls)
    assert not any(name == "docker_build" for name, _ in calls)
    assert not any(name == "docker_push" for name, _ in calls)
    assert any(name == "deploy" for name, _ in calls)
