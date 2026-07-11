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
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_adapters import BRANCHES, REPO_ID  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict the check to these branch names (default: all of BRANCHES).",
    )
    return parser


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
    args = build_parser().parse_args()
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
