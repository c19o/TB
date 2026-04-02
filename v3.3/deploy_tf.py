#!/usr/bin/env python3
"""Maintained machine-agnostic deploy engine for Savage22 timeframes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from deploy_manifest import MANIFEST_PATH, build_manifest
from deploy_profiles import env_defaults, load_timeframe_profile, requires_cupy, runtime_home_default
from gcs_shared_seed import build_client
from pipeline_contract import load_timeframe_contract, ordered_phase_names


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REMOTE_DIR_DEFAULT = "/workspace"
CURRENT_LINK_NAME = "current_v3.3"
REMOTE_REQUIRED_PIP_PACKAGES = [
    "numpy<2.3",
    "pandas<3.0",
    "google-cloud-storage",
    "lightgbm",
    "scikit-learn",
    "scipy",
    "ephem",
    "astropy",
    "pytz",
    "joblib",
    "pyarrow",
    "optuna",
    "hmmlearn",
    "numba",
    "tqdm",
    "pyyaml",
]


def _now_run_id(tf: str, instance_id: str | None) -> str:
    suffix = instance_id or "manual"
    return f"{tf}-{suffix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: list[str], *, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        check=True,
        capture_output=capture,
    )


def _which_or_default(name: str, default: str) -> str:
    return shutil.which(name) or default


def _resolve_vast_instance(instance_id: str) -> tuple[str, int]:
    vastai = _which_or_default("vastai", "vastai")
    raw = _run([vastai, "show", "instance", instance_id, "--raw"], capture=True).stdout.strip()
    payload = json.loads(raw)
    if isinstance(payload, list):
        payload = payload[0]
    ssh_host = payload.get("ssh_host") or payload.get("sshHost")
    ssh_port = payload.get("ssh_port") or payload.get("sshPort")
    if not ssh_host or not ssh_port:
        raise RuntimeError(f"Unable to resolve ssh host/port for instance {instance_id}")
    return str(ssh_host), int(ssh_port)


def _remote(ssh_bin: str, ssh_host: str, ssh_port: int, command: str) -> None:
    _run([ssh_bin, "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no", f"root@{ssh_host}", command])


def _upload(scp_bin: str, ssh_host: str, ssh_port: int, local_path: Path, remote_target: str) -> None:
    _run([scp_bin, "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no", str(local_path), f"root@{ssh_host}:{remote_target}"])


def _bundle_release(relpaths: list[str]) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="savage22_release_", suffix=".tgz", delete=False)
    handle.close()
    bundle_path = Path(handle.name)
    with tarfile.open(bundle_path, "w:gz") as tar:
        for relpath in relpaths:
            src = SCRIPT_DIR / relpath
            tar.add(src, arcname=relpath)
    return bundle_path


def _release_manifest_payload(
    *,
    run_id: str,
    tf: str,
    release_dir: str,
    run_dir: str,
    artifact_root: str,
    shared_db_root: str,
    heartbeat_path: str,
    seed_meta: dict,
    current_link: str | None = None,
) -> dict:
    payload = {
        "run_id": run_id,
        "tf": tf,
        "release_dir": release_dir,
        "run_dir": run_dir,
        "artifact_root": artifact_root,
        "shared_db_root": shared_db_root,
        "heartbeat_path": heartbeat_path,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deploy_profile": "contracts/deploy_profiles.json",
        "artifact_contract": "contracts/pipeline_contract.json",
        "shared_db_seed": seed_meta,
    }
    if current_link:
        payload["current_link"] = current_link
    return payload


def _shell_exports(env_map: dict[str, str]) -> str:
    parts = [f"export {key}={shlex.quote(value)}" for key, value in env_map.items()]
    return " && ".join(parts)


def _seed_metadata(bucket: str, prefix: str, key_file: str) -> dict:
    client, project_id = build_client(key_file)
    blob = client.bucket(bucket).blob(f"{prefix.strip('/')}/manifest.json")
    if not blob.exists():
        raise FileNotFoundError(f"Missing shared seed manifest: gs://{bucket}/{prefix.strip('/')}/manifest.json")
    manifest_text = blob.download_as_text()
    payload = json.loads(manifest_text)
    return {
        "project_id": project_id,
        "bucket": bucket,
        "prefix": prefix,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "file_count": len(payload.get("files", {})),
    }


def _install_remote_python_packages(
    ssh_bin: str,
    ssh_host: str,
    ssh_port: int,
    remote_release_dir: str,
    tf: str,
) -> None:
    package_map = {
        "google.cloud.storage": "google-cloud-storage",
        "lightgbm": "lightgbm",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "ephem": "ephem",
        "astropy": "astropy",
        "pytz": "pytz",
        "joblib": "joblib",
        "pandas": "pandas<3.0",
        "pyarrow": "pyarrow",
        "optuna": "optuna",
        "hmmlearn": "hmmlearn",
        "numba": "numba",
        "tqdm": "tqdm",
        "yaml": "pyyaml",
    }
    if requires_cupy(tf):
        package_map["cupy"] = "cupy-cuda12x"
    remote_script = (
        "import importlib, subprocess, sys\n"
        f"packages = {package_map!r}\n"
        "missing = []\n"
        "for module_name, package_name in packages.items():\n"
        "    try:\n"
        "        importlib.import_module(module_name)\n"
        "    except Exception:\n"
        "        missing.append(package_name)\n"
        "if missing:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])\n"
    )
    cmd = (
        f"cd {shlex.quote(remote_release_dir)} && "
        f"python - <<'PY'\n{remote_script}PY"
    )
    _remote(ssh_bin, ssh_host, ssh_port, cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", required=True, choices=["1w", "1d", "4h", "1h", "15m"])
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--ssh-host", default="")
    parser.add_argument("--ssh-port", type=int, default=0)
    parser.add_argument("--remote-dir", default=REMOTE_DIR_DEFAULT)
    parser.add_argument("--runtime-home", default=os.environ.get("SAVAGE22_RUNTIME_HOME", runtime_home_default()))
    parser.add_argument("--gcs-project-id", default=os.environ.get("GCS_PROJECT_ID", ""))
    parser.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET", ""))
    parser.add_argument("--gcs-prefix", default=os.environ.get("GCS_PREFIX", ""))
    parser.add_argument("--gcs-key-file", default=os.environ.get("GCS_KEY_PATH", ""))
    parser.add_argument("--session-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    tf = args.tf
    profile = load_timeframe_profile(tf)
    contract = load_timeframe_contract(tf)

    if not args.ssh_host and args.instance_id:
        args.ssh_host, args.ssh_port = _resolve_vast_instance(args.instance_id)
    if not args.dry_run and (not args.ssh_host or not args.ssh_port):
        raise SystemExit("Provide --ssh-host and --ssh-port, or --instance-id resolvable via vastai.")

    run_id = _now_run_id(tf, args.instance_id or None)
    remote_dir = args.remote_dir.rstrip("/")
    remote_releases = f"{remote_dir}/releases"
    remote_runs = f"{remote_dir}/runs"
    remote_artifacts = f"{remote_dir}/artifacts"
    remote_release_dir = f"{remote_releases}/v3.3_{run_id}"
    remote_staging_dir = f"{remote_release_dir}.staging"
    remote_run_dir = f"{remote_runs}/{run_id}"
    remote_artifact_root = f"{remote_artifacts}/{run_id}"
    remote_current_link = f"{remote_dir}/{CURRENT_LINK_NAME}"
    remote_heartbeat = f"{remote_run_dir}/cloud_run_{tf}_heartbeat.json"
    session_name = args.session_name or f"train_{tf}"

    manifest = build_manifest()
    _json_dump(Path(MANIFEST_PATH), manifest)
    release_files = sorted(manifest["files"].keys())
    if not args.dry_run and (not args.gcs_bucket or not args.gcs_prefix or not args.gcs_key_file):
        raise SystemExit("Maintained deploys require --gcs-bucket, --gcs-prefix, and --gcs-key-file.")
    seed_meta = {
        "project_id": args.gcs_project_id,
        "bucket": args.gcs_bucket,
        "prefix": args.gcs_prefix,
        "manifest_sha256": "",
        "file_count": 0,
    }
    if args.gcs_bucket and args.gcs_prefix and args.gcs_key_file:
        seed_meta = _seed_metadata(args.gcs_bucket, args.gcs_prefix, args.gcs_key_file)

    summary = {
        "tf": tf,
        "run_id": run_id,
        "warm_start_parent": profile.get("warm_start_parent"),
        "execution_mode": profile.get("execution_mode"),
        "same_machine_required": profile.get("same_machine_required", False),
        "phase_order": ordered_phase_names(tf),
        "machine_policy": profile.get("machine_policy"),
        "env_defaults": env_defaults(tf),
        "runtime_home": args.runtime_home,
        "release_file_count": len(release_files),
        "remote": {
            "ssh_host": args.ssh_host,
            "ssh_port": args.ssh_port,
            "release_dir": remote_release_dir,
            "staging_dir": remote_staging_dir,
            "run_dir": remote_run_dir,
            "artifact_root": remote_artifact_root,
            "current_link": remote_current_link,
        },
        "contract_path": "contracts/pipeline_contract.json",
        "deploy_profile_path": "contracts/deploy_profiles.json",
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    ssh_bin = _which_or_default("ssh", "ssh")
    scp_bin = _which_or_default("scp", "scp")

    bundle_path = _bundle_release(release_files)
    local_release_manifest = Path(tempfile.NamedTemporaryFile(prefix="savage22_release_manifest_", suffix=".json", delete=False).name)
    _json_dump(
        local_release_manifest,
        _release_manifest_payload(
            run_id=run_id,
            tf=tf,
            release_dir=remote_staging_dir,
            run_dir=remote_run_dir,
            artifact_root=remote_artifact_root,
            shared_db_root=remote_dir,
            heartbeat_path=remote_heartbeat,
            seed_meta=seed_meta,
        ),
    )

    remote_bundle = f"{remote_run_dir}/{bundle_path.name}"
    try:
        _remote(ssh_bin, args.ssh_host, args.ssh_port, f"mkdir -p {shlex.quote(remote_runs)} {shlex.quote(remote_artifacts)} {shlex.quote(remote_releases)} {shlex.quote(remote_run_dir)} {shlex.quote(remote_artifact_root)}")
        _upload(scp_bin, args.ssh_host, args.ssh_port, bundle_path, remote_bundle)
        _remote(
            ssh_bin,
            args.ssh_host,
            args.ssh_port,
            f"mkdir -p {shlex.quote(remote_staging_dir)} && tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(remote_staging_dir)} && rm -f {shlex.quote(remote_bundle)}",
        )
        _install_remote_python_packages(ssh_bin, args.ssh_host, args.ssh_port, remote_staging_dir, tf)
        _upload(scp_bin, args.ssh_host, args.ssh_port, local_release_manifest, f"{remote_run_dir}/release_manifest.json")

        if args.gcs_key_file and args.gcs_bucket and args.gcs_prefix:
            key_path = Path(args.gcs_key_file).expanduser().resolve()
            remote_key = f"{remote_run_dir}/gcs_service_account.json"
            _upload(scp_bin, args.ssh_host, args.ssh_port, key_path, remote_key)
            gcs_cmd = (
                f"cd {shlex.quote(remote_staging_dir)} && "
                f"python gcs_shared_seed.py download "
                f"--bucket {shlex.quote(args.gcs_bucket)} "
                f"--prefix {shlex.quote(args.gcs_prefix)} "
                f"--key-file {shlex.quote(remote_key)} "
                f"--dest {shlex.quote(remote_dir)} && "
                f"rm -f {shlex.quote(remote_key)}"
            )
            _remote(ssh_bin, args.ssh_host, args.ssh_port, gcs_cmd)

        remote_env = {
            "SAVAGE22_RUN_ID": run_id,
            "SAVAGE22_ARTIFACT_DIR": remote_artifact_root,
            "SAVAGE22_RUN_DIR": remote_run_dir,
            "SAVAGE22_DB_DIR": remote_dir,
            "SAVAGE22_V1_DIR": remote_dir,
            "V30_DATA_DIR": remote_artifact_root,
        }
        remote_env.update(env_defaults(tf))
        env_export = _shell_exports(remote_env)

        preflight_cmd = (
            f"cd {shlex.quote(remote_staging_dir)} && {env_export} && "
            f"python -X utf8 deploy_verify.py --tf {shlex.quote(tf)} --allow-staged-release && "
            f"python -X utf8 validate.py --tf {shlex.quote(tf)} --cloud"
        )
        _remote(ssh_bin, args.ssh_host, args.ssh_port, preflight_cmd)

        promote_cmd = (
            f"rm -rf {shlex.quote(remote_release_dir)} && "
            f"mv {shlex.quote(remote_staging_dir)} {shlex.quote(remote_release_dir)} && "
            f"ln -sfn {shlex.quote(remote_release_dir)} {shlex.quote(remote_current_link)}"
        )
        _remote(ssh_bin, args.ssh_host, args.ssh_port, promote_cmd)

        final_manifest = _release_manifest_payload(
            run_id=run_id,
            tf=tf,
            release_dir=remote_release_dir,
            run_dir=remote_run_dir,
            artifact_root=remote_artifact_root,
            shared_db_root=remote_dir,
            heartbeat_path=remote_heartbeat,
            seed_meta=seed_meta,
            current_link=remote_current_link,
        )
        _json_dump(local_release_manifest, final_manifest)
        _upload(scp_bin, args.ssh_host, args.ssh_port, local_release_manifest, f"{remote_run_dir}/release_manifest.json")

        if not args.no_launch:
            launch_cmd = (
                f"mkdir -p {shlex.quote(remote_run_dir)}/logs && "
                f"tmux kill-session -t {shlex.quote(session_name)} >/dev/null 2>&1 || true && "
                f"tmux new-session -d -s {shlex.quote(session_name)} "
                f"\"cd {shlex.quote(remote_release_dir)} && {env_export} && python -X utf8 -u cloud_run_tf.py --tf {shlex.quote(tf)} 2>&1 | tee {shlex.quote(remote_run_dir)}/logs/console_{tf}.log\""
            )
            _remote(ssh_bin, args.ssh_host, args.ssh_port, launch_cmd)
    finally:
        try:
            bundle_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            local_release_manifest.unlink(missing_ok=True)
        except Exception:
            pass

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
