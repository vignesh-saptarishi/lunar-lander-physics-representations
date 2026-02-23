#!/usr/bin/env python3
"""Render recorded clips to clean mp4 videos + companion JSON metadata.

Takes a directory of .npz clips (from record_clips.py) and renders each
as a clean game-view-only mp4 (no annotation panel). Each clip gets a
companion .json file with physics config, outcome, behavioral metrics.

Usage:
    python lunar_lander/scripts/render_clips.py \
        --input-dir /path/to/clips/ \
        --output-dir /path/to/rendered/ \
        --variant labeled \
        --condition full-variation-easy
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.clip_recording import render_clean_clip


def main():
    parser = argparse.ArgumentParser(
        description="Render recorded clips to clean mp4 + companion JSON.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with .npz clips (from record_clips.py)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for mp4 + JSON files"
    )
    parser.add_argument("--variant", default="", help="Variant label for metadata")
    parser.add_argument("--condition", default="", help="Condition label for metadata")
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        sys.exit(1)

    print(f"Rendering {len(npz_files)} clips from {input_dir}")
    print(f"  Variant: {args.variant}")
    print(f"  Condition: {args.condition}")
    print()

    for npz_path in npz_files:
        mp4_name = npz_path.stem + ".mp4"
        mp4_path = output_dir / mp4_name

        mp4_out, json_out = render_clean_clip(
            npz_path,
            mp4_path,
            fps=args.fps,
            variant=args.variant,
            condition=args.condition,
        )
        print(f"  {npz_path.name} -> {mp4_out.name} + {json_out.name}")

    print(f"\nDone. {len(npz_files)} clips rendered to {output_dir}/")


if __name__ == "__main__":
    main()
