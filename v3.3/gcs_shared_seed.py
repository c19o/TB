#!/usr/bin/env python3
"""
Sync Savage22 shared DB seed files to/from Google Cloud Storage.

Usage:
  python gcs_shared_seed.py upload --bucket <bucket> --prefix <prefix> --key-file <json> --project-root <root>
  python gcs_shared_seed.py download --bucket <bucket> --prefix <prefix> --key-file <json> --dest <dir>
  python gcs_shared_seed.py manifest --bucket <bucket> --prefix <prefix> --key-file <json> --manifest-out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REQUIRED_FILES = [
    "astrology_full.db",
    "btc_prices.db",
    "ephemeris_cache.db",
    "fear_greed.db",
    "funding_rates.db",
    "google_trends.db",
    "llm_cache.db",
    "macro_data.db",
    "multi_asset_prices.db",
    "news_articles.db",
    "onchain_data.db",
    "open_interest.db",
    "space_weather.db",
    "sports_results.db",
    "tweets.db",
    "v2_signals.db",
    "kp_history_gfz.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_client(key_file: str):
    from google.cloud import storage
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(key_file)
    return storage.Client(project=creds.project_id, credentials=creds), creds.project_id


def manifest_matches(remote_manifest: dict | None, local_manifest: dict) -> bool:
    if not remote_manifest:
        return False
    return (
        remote_manifest.get("project_id") == local_manifest.get("project_id")
        and remote_manifest.get("bucket") == local_manifest.get("bucket")
        and remote_manifest.get("prefix") == local_manifest.get("prefix")
        and remote_manifest.get("files") == local_manifest.get("files")
    )


def resolve_source_file(project_root: Path, name: str) -> Path:
    candidates = [
        project_root / "v3.3" / name,
        project_root / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required seed file not found: {name}")


def build_manifest(project_root: Path, project_id: str, bucket_name: str, prefix: str) -> tuple[dict, dict[str, Path]]:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_id": project_id,
        "bucket": bucket_name,
        "prefix": prefix,
        "files": {},
    }
    source_files: dict[str, Path] = {}
    for name in REQUIRED_FILES:
        src = resolve_source_file(project_root, name)
        source_files[name] = src
        manifest["files"][name] = {
            "size": src.stat().st_size,
            "sha256": sha256_file(src),
        }
    return manifest, source_files


def upload_seed(args) -> int:
    project_root = Path(args.project_root).resolve()
    client, project_id = build_client(args.key_file)
    bucket = client.bucket(args.bucket)
    prefix = args.prefix.strip("/")
    manifest_blob = bucket.blob(f"{prefix}/manifest.json")
    remote_manifest = None
    if manifest_blob.exists():
        try:
            remote_manifest = json.loads(manifest_blob.download_as_text())
        except Exception:
            remote_manifest = None

    manifest, source_files = build_manifest(project_root, project_id, args.bucket, prefix)
    remote_files = (remote_manifest or {}).get("files", {})
    uploads = 0
    for name, meta in manifest["files"].items():
        if remote_files.get(name) == meta:
            print(f"OK {name}")
            continue
        src = source_files[name]
        blob = bucket.blob(f"{prefix}/{name}")
        blob.upload_from_filename(str(src))
        uploads += 1
        print(f"UPLOADED {name} ({meta['size'] / 1e6:.1f} MB)")

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    if uploads > 0 or not manifest_matches(remote_manifest, manifest):
        manifest_blob.upload_from_string(manifest_json, content_type="application/json")
        print(f"Manifest uploaded: gs://{args.bucket}/{prefix}/manifest.json")
    else:
        print(f"Manifest unchanged: gs://{args.bucket}/{prefix}/manifest.json")
    if args.manifest_out:
        out_path = Path(args.manifest_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(manifest_json, encoding="utf-8")
    return 0


def download_seed(args) -> int:
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    client, _project_id = build_client(args.key_file)
    bucket = client.bucket(args.bucket)
    prefix = args.prefix.strip("/")

    manifest_blob = bucket.blob(f"{prefix}/manifest.json")
    if not manifest_blob.exists():
        raise FileNotFoundError(f"Missing GCS seed manifest: gs://{args.bucket}/{prefix}/manifest.json")
    manifest = json.loads(manifest_blob.download_as_text())

    for name, meta in manifest.get("files", {}).items():
        local_path = dest / name
        needs_download = True
        if local_path.exists():
            if local_path.stat().st_size == meta.get("size") and sha256_file(local_path) == meta.get("sha256"):
                needs_download = False
        if needs_download:
            blob = bucket.blob(f"{prefix}/{name}")
            blob.download_to_filename(str(local_path))
            print(f"DOWNLOADED {name}")
        else:
            print(f"OK {name}")

        if local_path.stat().st_size != meta.get("size") or sha256_file(local_path) != meta.get("sha256"):
            raise RuntimeError(f"Downloaded file failed verification: {name}")
    return 0


def write_manifest(args) -> int:
    out_path = Path(args.manifest_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client, _project_id = build_client(args.key_file)
    bucket = client.bucket(args.bucket)
    prefix = args.prefix.strip("/")

    manifest_blob = bucket.blob(f"{prefix}/manifest.json")
    if not manifest_blob.exists():
        raise FileNotFoundError(f"Missing GCS seed manifest: gs://{args.bucket}/{prefix}/manifest.json")

    manifest_text = manifest_blob.download_as_text()
    # Validate the JSON before writing it out so the caller can trust the file.
    json.loads(manifest_text)
    out_path.write_text(manifest_text, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload")
    up.add_argument("--bucket", required=True)
    up.add_argument("--prefix", required=True)
    up.add_argument("--key-file", required=True)
    up.add_argument("--project-root", default=str(PROJECT_ROOT))
    up.add_argument("--manifest-out", default="")

    down = sub.add_parser("download")
    down.add_argument("--bucket", required=True)
    down.add_argument("--prefix", required=True)
    down.add_argument("--key-file", required=True)
    down.add_argument("--dest", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--bucket", required=True)
    manifest.add_argument("--prefix", required=True)
    manifest.add_argument("--key-file", required=True)
    manifest.add_argument("--manifest-out", required=True)

    args = parser.parse_args()
    if args.cmd == "upload":
        return upload_seed(args)
    if args.cmd == "download":
        return download_seed(args)
    return write_manifest(args)


if __name__ == "__main__":
    sys.exit(main())
