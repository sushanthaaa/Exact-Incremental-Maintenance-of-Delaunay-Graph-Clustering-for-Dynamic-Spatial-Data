#!/usr/bin/env python3
"""Verify and restore the archived paper inputs; no network access required."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def restore(root: Path, verify_only: bool = False) -> int:
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot manifest version")
    pending = []
    # Validate every file before writing any of them.
    for entry in manifest["snapshots"]:
        packed = inside(root, entry["archive"]).read_bytes()
        if digest(packed) != entry["archive_sha256"]:
            raise ValueError(f"Archive checksum failed: {entry['archive']}")
        raw = gzip.decompress(packed)
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"Input checksum failed: {entry['destination']}")
        target = inside(root, entry["destination"])
        if target.exists():
            if digest(target.read_bytes()) != entry["sha256"]:
                raise ValueError(f"Existing input differs; preserve or move it first: {target}")
        else:
            pending.append((target, raw))
    if not verify_only:
        for target, raw in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids replacing a file created after validation.
            with target.open("xb") as handle:
                handle.write(raw)
    return len(manifest["snapshots"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="Check archives and any existing inputs without writing")
    args = parser.parse_args()
    count = restore(ROOT, args.verify_only)
    print(f"Verified {count} snapshots" + ("" if args.verify_only else "; inputs restored"))


if __name__ == "__main__":
    main()
