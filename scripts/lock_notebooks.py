#!/usr/bin/env -S uv run --with tomllib-stubs python3
"""Pin (lock) or unpin (unlock) PEP 723 inline script dependencies in marimo notebooks.

Lock mode (default):
  For each notebook, resolves unpinned deps via `uv lock --script`, then
  rewrites the inline metadata with exact versions using `uv add --script`.
  Infrastructure deps (marimo, python-dotenv) stay unpinned.

Unlock mode (--unlock):
  Strips version pins from inline deps, restoring them to bare package names.

Usage:
  python scripts/lock_notebooks.py notebooks/nb*.py
  python scripts/lock_notebooks.py --unlock notebooks/nb*.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# Deps that should stay unpinned - they're infrastructure, not analysis deps.
SKIP_PINNING = {"marimo", "python-dotenv"}


def extract_pep723_deps(path: Path) -> list[str]:
    """Extract dependency names from a PEP 723 inline script metadata block."""
    text = path.read_text()
    match = re.search(r"^# /// script\s*\n(.*?)^# ///\s*$", text, re.MULTILINE | re.DOTALL)
    if not match:
        return []

    # Reconstruct the TOML by stripping the "# " prefix from each line
    toml_lines = []
    for line in match.group(1).splitlines():
        if line.startswith("# "):
            toml_lines.append(line[2:])
        elif line.strip() == "#":
            toml_lines.append("")
        else:
            toml_lines.append(line)

    toml_text = "\n".join(toml_lines)
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return []
    return data.get("dependencies", [])


def normalize_name(dep: str) -> str:
    """Extract the bare package name from a dependency specifier."""
    return re.split(r"[><=!~\[]", dep)[0].strip().lower().replace("-", "-")


def resolve_versions(notebook: Path) -> dict[str, str]:
    """Run uv lock --script, parse the lockfile, return {name: version} for top-level deps."""
    lockfile = notebook.parent / f"{notebook.name}.lock"

    try:
        result = subprocess.run(
            ["uv", "lock", "--script", str(notebook)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  WARNING: uv lock failed: {result.stderr.strip()}", file=sys.stderr)
            return {}

        if not lockfile.exists():
            print(f"  WARNING: lockfile not created for {notebook.name}", file=sys.stderr)
            return {}

        with open(lockfile, "rb") as f:
            lock = tomllib.load(f)

        # Get top-level requirement names
        top_level = {r["name"] for r in lock.get("manifest", {}).get("requirements", [])}

        # Map name -> version from the package list
        versions = {}
        for pkg in lock.get("package", []):
            if pkg["name"] in top_level:
                versions[pkg["name"]] = pkg["version"]

        return versions
    finally:
        # Always clean up the sidecar lockfile
        if lockfile.exists():
            lockfile.unlink()


def lock_notebook(notebook: Path, dry_run: bool = False) -> bool:
    """Pin deps in a single notebook. Returns True if changes were made."""
    deps = extract_pep723_deps(notebook)
    if not deps:
        print(f"  SKIP {notebook.name}: no PEP 723 deps")
        return False

    # Figure out which deps need pinning
    unpinned = []
    for dep in deps:
        name = normalize_name(dep)
        if name in SKIP_PINNING:
            continue
        # Already has an exact pin (==)?
        if "==" in dep:
            continue
        unpinned.append(name)

    if not unpinned:
        print(f"  SKIP {notebook.name}: all deps already pinned (or skipped)")
        return False

    # Resolve versions
    versions = resolve_versions(notebook)
    if not versions:
        print(f"  SKIP {notebook.name}: could not resolve versions")
        return False

    # Build the list of pinned deps to add
    pins = []
    for name in unpinned:
        if name in versions:
            pins.append(f"{name}=={versions[name]}")
        else:
            print(f"  WARNING: no resolved version for {name} in {notebook.name}")

    if not pins:
        print(f"  SKIP {notebook.name}: no versions to pin")
        return False

    if dry_run:
        print(f"  DRY-RUN {notebook.name}: would pin {', '.join(pins)}")
        return True

    # Use uv add --script to rewrite the inline metadata
    cmd = ["uv", "add", "--script", str(notebook)] + pins
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # uv add --script also creates a .lock file - clean it up
    lockfile = notebook.parent / f"{notebook.name}.lock"
    if lockfile.exists():
        lockfile.unlink()

    if result.returncode != 0:
        print(f"  ERROR {notebook.name}: uv add failed: {result.stderr.strip()}")
        return False

    print(f"  LOCKED {notebook.name}: {', '.join(pins)}")
    return True


def unlock_notebook(notebook: Path, dry_run: bool = False) -> bool:
    """Strip version pins from a single notebook via text rewrite. Returns True if changes were made."""
    text = notebook.read_text()

    # Find the PEP 723 block
    match = re.search(r"(^# /// script\s*\n)(.*?)(^# ///\s*$)", text, re.MULTILINE | re.DOTALL)
    if not match:
        print(f"  SKIP {notebook.name}: no PEP 723 block")
        return False

    block = match.group(2)

    # Strip version specifiers from dep lines, but skip SKIP_PINNING deps
    # Match lines like: #     "numpy==2.4.6",  or  #     "pooch>=1.8.2,<2",
    def strip_pin(m: re.Match) -> str:
        indent = m.group(1)
        name = m.group(2)
        trailing = m.group(3)
        if name.lower().replace("-", "-") in SKIP_PINNING:
            return m.group(0)  # leave infrastructure deps alone
        return f'{indent}"{name}"{trailing}'

    new_block = re.sub(
        r'(#\s+)"([a-zA-Z0-9_-]+)[^"]*"(,?\s*)',
        strip_pin,
        block,
    )

    if new_block == block:
        print(f"  SKIP {notebook.name}: no pins to strip")
        return False

    # Count what changed
    changed_deps = []
    for old_line, new_line in zip(block.splitlines(), new_block.splitlines()):
        if old_line != new_line:
            dep_match = re.search(r'"([^"]+)"', old_line)
            if dep_match:
                changed_deps.append(normalize_name(dep_match.group(1)))

    if dry_run:
        print(f"  DRY-RUN {notebook.name}: would unpin {', '.join(changed_deps)}")
        return True

    new_text = text[: match.start(2)] + new_block + text[match.end(2) :]
    notebook.write_text(new_text)

    print(f"  UNLOCKED {notebook.name}: {', '.join(changed_deps)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pin/unpin PEP 723 inline deps")
    parser.add_argument("notebooks", nargs="+", type=Path, help="Notebook .py files")
    parser.add_argument("--unlock", action="store_true", help="Strip pins instead of adding them")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    args = parser.parse_args()

    action = unlock_notebook if args.unlock else lock_notebook
    verb = "Unlocking" if args.unlock else "Locking"

    print(f"{verb} {len(args.notebooks)} notebook(s)...")
    changed = 0
    for nb in sorted(args.notebooks):
        if not nb.exists():
            print(f"  SKIP {nb}: file not found")
            continue
        if action(nb, dry_run=args.dry_run):
            changed += 1

    print(f"\nDone: {changed}/{len(args.notebooks)} notebook(s) {'would be ' if args.dry_run else ''}changed.")


if __name__ == "__main__":
    main()
