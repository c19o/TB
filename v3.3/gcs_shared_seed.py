#!/usr/bin/env python3
"""
Sync Savage22 shared DB seed files to/from Google Cloud Storage.

Usage:
  python gcs_shared_seed.py upload --bucket <bucket> --prefix <prefix> --key-file <json> --project-root <root>
  python gcs_shared_seed.py download --bucket <bucket> --prefix <prefix> --key-file <json> --dest <dir>
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


def resolve_source_file(project_root: Path, name: str) -> Path:
    candidates = [
        project_root / "v3.3" / name,
        project_root / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required seed file not found: {name}")


def upload_seed(args) -> int:
    project_root = Path(args.project_root).resolve()
    client, project_id = build_client(args.key_file)
    bucket = client.bucket(args.bucket)
    prefix = args.prefix.strip("/")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_id": project_id,
        "bucket": args.bucket,
        "prefix": prefix,
        "files": {},
    }

    for name in REQUIRED_FILES:
        src = resolve_source_file(project_root, name)
        blob = bucket.blob(f"{prefix}/{name}")
        digest = sha256_file(src)
        size = src.stat().st_size
        blob.upload_from_filename(str(src))
        manifest["files"][name] = {
            "size": size,
            "sha256": digest,
        }
        print(f"UPLOADED {name} ({size / 1e6:.1f} MB)")

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_blob = bucket.blob(f"{prefix}/manifest.json")
    manifest_blob.upload_from_string(manifest_json, content_type="application/json")
    if args.manifest_out:
        out_path = Path(args.manifest_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(manifest_json, encoding="utf-8")
    print(f"Manifest uploaded: gs://{args.bucket}/{prefix}/manifest.json")
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

    args = parser.parse_args()
    if args.cmd == "upload":
        return upload_seed(args)
    return download_seed(args)


if __name__ == "__main__":
    sys.exit(main())
