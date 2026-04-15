#!/usr/bin/env python3
"""Download the SyntheWorld payload from Hugging Face when only Git LFS pointers exist.

This helper is intentionally small so setup.sh can invoke it without running
the full evaluation pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-root",
        required=True,
        help="Path to Dataset/synthetic_dataset",
    )
    args = parser.parse_args()

    synthetic_root = Path(args.synthetic_root).expanduser().resolve()
    syntheworld_root = synthetic_root / "SyntheWorld"
    syntheworld_root.mkdir(parents=True, exist_ok=True)

    try:
        from E1_batch_texture_eval import ensure_syntheworld_payload
    except Exception as exc:  # pragma: no cover - import-time dependency guard
        print(f"[syntheworld] unable to import helper: {exc}")
        return 1

    ok = ensure_syntheworld_payload(syntheworld_root)
    if ok:
        print(f"[syntheworld] ready: {syntheworld_root}")
        return 0

    print("[syntheworld] nothing to download or download failed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())