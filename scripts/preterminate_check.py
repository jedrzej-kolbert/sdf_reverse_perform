#!/usr/bin/env python
"""Go/no-go check before terminating a billed Lambda GPU instance.

Lambda has no "stopped, disk-preserved" state -- only running (billed) and
terminated (billing stops, disk wiped). Termination is irreversible, so this
verifies every expected adapter in `upload_adapters.BRANCHES` is durably off
the instance (present locally, or already pushed to the HF repo) before
printing GO. It does not itself terminate anything.

Usage:
  uv run python scripts/preterminate_check.py
  uv run python scripts/preterminate_check.py --only insert-r5-8000 qwen17-28088
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_adapters import BRANCHES, REPO_ID  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Builds registry-based and explicit-checksum verification arguments.

    Returns:
        The configured command-line parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict the check to these branch names (default: all of BRANCHES).",
    )
    parser.add_argument("--result-root", type=Path, help="Durable local result directory.")
    parser.add_argument("--checksum-manifest", type=Path,
                        help="JSON list of path/bytes/sha256 records from the instance.")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Allow an empty inventory after a failed boot/setup only.")
    return parser


def verify_checksum_manifest(result_root: Path, manifest_path: Path,
                             allow_empty: bool = False) -> list[str]:
    """Verifies an explicit instance inventory against durable local files.

    Args:
        result_root: Directory outside the billed instance containing pulled results.
        manifest_path: JSON list of relative filenames, sizes, and SHA-256 digests.
        allow_empty: Permit no artifacts only when setup failed before work started.

    Returns:
        Failure descriptions, or an empty list when every record matches.

    Raises:
        ValueError: The inventory is malformed or includes an unsafe relative path.
    """
    records = json.loads(manifest_path.read_text())
    if not isinstance(records, list) or (not records and not allow_empty):
        raise ValueError("Expected a non-empty JSON artifact inventory")
    root = result_root.resolve()
    failures: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("Expected path/bytes/sha256 inventory records")
        relative = Path(record["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Unsafe inventory path: {relative}")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            failures.append(f"Missing or truncated: {relative}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != record["sha256"]:
            failures.append(f"Checksum mismatch: {relative}")
    return failures


def branch_on_hub(api: HfApi, branch: str) -> bool:
    """Checks whether a branch exists on the HF repo with adapter weights present.

    Args:
        api: Authenticated HfApi client.
        branch: HF Hub branch name to check.

    Returns:
        True if the branch exists and contains an adapter weights file.
    """
    try:
        files = api.list_repo_files(REPO_ID, revision=branch)
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return False
    return any(f.endswith((".safetensors", ".bin")) for f in files)


def main() -> None:
    """Checks registered adapters or an explicit durable-local file inventory."""
    args = build_parser().parse_args()
    if args.result_root is not None or args.checksum_manifest is not None:
        if args.result_root is None or args.checksum_manifest is None:
            raise SystemExit("Both --result-root and --checksum-manifest are required")
        failures = verify_checksum_manifest(args.result_root, args.checksum_manifest, args.allow_empty)
        if failures:
            for failure in failures:
                print(f"NO-GO: {failure}")
            raise SystemExit(1)
        print("GO: every inventory file is durably local with its instance checksum.")
        return
    branches = {k: v for k, v in BRANCHES.items() if args.only is None or k in args.only}
    if not branches:
        raise SystemExit(f"No matching branches for --only {args.only}")

    api = HfApi()
    missing: list[str] = []

    for branch, folder_str in branches.items():
        folder = Path(folder_str)
        local_ok = folder.is_dir() and any(
            (folder / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
        remote_ok = branch_on_hub(api, branch)

        if local_ok or remote_ok:
            where = "local" if local_ok else ""
            where += ("+" if where and remote_ok else "") + ("hub" if remote_ok else "")
            print(f"OK    {branch}: {where}")
        else:
            print(f"MISS  {branch}: not found locally ({folder}) or on {REPO_ID}@{branch}")
            missing.append(branch)

    print()
    if missing:
        print(f"NO-GO: {len(missing)} adapter(s) missing everywhere -- do not terminate:")
        for branch in missing:
            print(f"  - {branch}")
        raise SystemExit(1)

    print("GO: every checked adapter is durably local or on the Hub. Safe to terminate.")


if __name__ == "__main__":
    main()
