#!/usr/bin/env -S uv run python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "python-dotenv"]
# ///
"""Upload files to a Zenodo deposit and print pooch-ready entries for nb43.

Usage:
    # Upload to default deposit (idempotent, skips unchanged files)
    uv run scripts/zenodo_upload.py data/external/file1.json data/external/file2.parquet

    # Create a new deposit and upload
    uv run scripts/zenodo_upload.py --new data/external/file1.json

    # Upload and publish
    uv run scripts/zenodo_upload.py --publish data/external/file1.json

    # Create a new version and replace a file
    uv run scripts/zenodo_upload.py --new-version 20388568 --replace data/external/file1.json

    # Apply metadata from JSON file
    uv run scripts/zenodo_upload.py --metadata metadata.json

    # Dry run: just show what would be uploaded
    uv run scripts/zenodo_upload.py --dry-run data/external/file1.json

Requires ZENODO_TOKEN env var (with deposit:write and deposit:actions scopes).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ZENODO_API = "https://zenodo.org/api"
DEPOSIT_ID = "20388583"


def file_hashes(path: Path) -> tuple[str, str]:
    """Compute MD5 and SHA-256 in a single read pass."""
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def get_token() -> str:
    load_dotenv()
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        print("Error: ZENODO_TOKEN not set (check env or .env)", file=sys.stderr)
        sys.exit(1)
    return token


def _api(method: str, url: str, token: str, data: dict | None = None, timeout: int = 30) -> dict:
    r = requests.request(
        method,
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=data if data is not None else ({} if method == "POST" else None),
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def upload_file(bucket_url: str, filepath: Path, token: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with open(filepath, "rb") as f:
                r = requests.put(
                    f"{bucket_url}/{filepath.name}",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"},
                    data=f,
                    timeout=600,
                )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries}: {e}", file=sys.stderr)
            else:
                raise
    return {}


def resolve_latest_id(deposit_id: str) -> str:
    r = requests.get(f"https://zenodo.org/records/{deposit_id}/latest", allow_redirects=False, timeout=30)
    if r.status_code in (301, 302):
        return r.headers.get("Location", "").rstrip("/").split("/")[-1]
    return deposit_id


def main():
    parser = argparse.ArgumentParser(description="Upload files to Zenodo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--deposit-id", default=DEPOSIT_ID, help=f"Draft deposit ID (default: {DEPOSIT_ID})")
    group.add_argument("--new", action="store_true", help="Create a new deposit")
    group.add_argument("--new-version", metavar="ID", help="Create new version of published deposit")
    parser.add_argument("--metadata", help="JSON file with deposit metadata")
    parser.add_argument("--replace", action="store_true", help="Delete existing files before uploading (for new versions)")
    parser.add_argument("--publish", action="store_true", help="Publish after upload")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("files", nargs="*", type=Path, help="Files to upload")
    args = parser.parse_args()

    token = get_token()
    deposit_id = args.deposit_id

    if args.new:
        if args.dry_run:
            print("Would create new deposit")
        else:
            dep = _api("POST", f"{ZENODO_API}/deposit/depositions", token)
            deposit_id = str(dep["id"])
            print(f"Created deposit {deposit_id} (DOI: {dep['metadata']['prereserve_doi']['doi']})")
    elif args.new_version:
        if args.dry_run:
            print(f"Would create new version of {args.new_version}")
        else:
            latest_id = resolve_latest_id(args.new_version)
            dep = _api("POST", f"{ZENODO_API}/deposit/depositions/{latest_id}/actions/newversion", token)
            deposit_id = str(dep["id"])
            print(f"Created new version {deposit_id}")

    if args.metadata and not args.dry_run:
        meta = json.loads(Path(args.metadata).read_text())
        _api("PUT", f"{ZENODO_API}/deposit/depositions/{deposit_id}", token, meta)
        print(f"Applied metadata from {args.metadata}")

    if args.files:
        remote_md5s: dict[str, str] = {}
        remote_file_ids: dict[str, str] = {}
        bucket_url = ""
        if not args.dry_run:
            dep = _api("GET", f"{ZENODO_API}/deposit/depositions/{deposit_id}", token)
            bucket_url = dep["links"]["bucket"]
            for f in dep.get("files", []):
                name = f.get("key") or f.get("filename")
                checksum = f.get("checksum", "")
                if name:
                    remote_md5s[name] = checksum if checksum.startswith("md5:") else f"md5:{checksum}"
                    remote_file_ids[name] = f["id"]

        hashes: dict[str, tuple[str, str]] = {}
        valid_files: list[Path] = []
        for filepath in args.files:
            if not filepath.exists():
                print(f"  SKIP {filepath} (not found)", file=sys.stderr)
                continue

            valid_files.append(filepath)
            md5, sha = file_hashes(filepath)
            hashes[filepath.name] = (md5, sha)

            if args.dry_run:
                action = "replace" if args.replace and filepath.name in remote_md5s else "upload"
                print(f"  Would {action} {filepath.name} ({filepath.stat().st_size:,} bytes, sha256:{sha[:16]}...)")
                continue

            if not args.replace and remote_md5s.get(filepath.name) == f"md5:{md5}":
                print(f"  SKIP {filepath.name} (unchanged)")
            else:
                if args.replace and filepath.name in remote_file_ids:
                    requests.delete(
                        f"{ZENODO_API}/deposit/depositions/{deposit_id}/files/{remote_file_ids[filepath.name]}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=30,
                    )
                print(f"  Uploading {filepath.name} ({filepath.stat().st_size:,} bytes)...", end=" ", flush=True)
                upload_file(bucket_url, filepath, token)
                print("done")

        print()
        print("# Pooch-ready entries for nb43 EXTERNAL_FILES:")
        for filepath in valid_files:
            _, sha = hashes[filepath.name]
            url = f"https://zenodo.org/records/{deposit_id}/files/{filepath.name}?download=1"
            print(f'        "{url}": (')
            print(f'            "{filepath.name}",')
            print(f'            "{sha}",')
            print(f"        ),")

    if args.publish and not args.dry_run:
        result = _api("POST", f"{ZENODO_API}/deposit/depositions/{deposit_id}/actions/publish", token)
        print(f"\nPublished: https://zenodo.org/records/{result['id']}")
        print(f"DOI: {result['doi']}")

    if not args.publish and not args.dry_run:
        print(f"\nDraft: https://zenodo.org/deposit/{deposit_id}")
        print("Add --publish to publish")


if __name__ == "__main__":
    main()
