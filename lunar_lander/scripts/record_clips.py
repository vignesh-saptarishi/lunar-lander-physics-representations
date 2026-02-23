#!/usr/bin/env python3
"""Record video clips from selected prototypical episodes.

Takes a selection manifest (JSON from select_prototypical_episodes.py)
and the agent checkpoint. For each selected episode, extracts the physics
config from the .npz metadata, then records a new episode with that exact
physics config — capturing rgb_frames for video rendering.

Usage:
    python lunar_lander/scripts/record_clips.py \
        --manifest selected.json \
        --checkpoint-dir /path/to/agent/s42/ \
        --output-dir /path/to/clips/

    # For corruption clips (same agent, corrupted inputs):
    python lunar_lander/scripts/record_clips.py \
        --manifest selected.json \
        --checkpoint-dir /path/to/agent/s42/ \
        --output-dir /path/to/clips/ \
        --corruption zero
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.clip_recording import extract_physics_config, record_clip


def main():
    parser = argparse.ArgumentParser(
        description="Record video clips from selected prototypical episodes.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Selection manifest JSON (from select_prototypical_episodes.py)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Agent checkpoint directory (model.zip, config.json, etc.)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for recorded .npz clips"
    )
    parser.add_argument(
        "--corruption",
        default=None,
        choices=["zero", "shuffle", "mean", "noise"],
        help="Optional corruption mode for labeled agents",
    )
    parser.add_argument(
        "--corruption-sigma",
        type=float,
        default=0.1,
        help="Noise sigma (only for --corruption noise)",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to env seed for terrain variation",
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["labeled", "blind", "history"],
        help="Override variant (default: read from checkpoint config.json)",
    )

    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    episodes = manifest["episodes"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Variant priority: CLI flag > checkpoint config.json > manifest.
    variant = args.variant
    if not variant:
        config_path = Path(args.checkpoint_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                train_config = json.load(f)
            variant = train_config.get("variant", "labeled")
        else:
            variant = manifest.get("variant", "labeled")

    print(f"Recording {len(episodes)} clips from {args.manifest}")
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Variant: {variant}")
    if args.corruption:
        print(f"  Corruption: {args.corruption}")
    print()

    for i, ep in enumerate(episodes):
        npz_path = ep["npz_path"]
        physics_config = extract_physics_config(npz_path)

        # Use the original episode's seed for terrain reproducibility.
        # Falls back to sequential index + offset if seed not in metadata.
        import numpy as np

        orig_data = np.load(npz_path, allow_pickle=False)
        orig_meta = json.loads(str(orig_data["metadata_json"]))
        episode_seed = orig_meta.get("seed", i) + args.seed_offset

        clip_name = f"clip_{i:03d}"
        output_path = output_dir / f"{clip_name}.npz"

        print(
            f"  [{i+1}/{len(episodes)}] Recording {clip_name} "
            f"(gravity={physics_config.gravity:.1f}, "
            f"seed={episode_seed}, "
            f"twr={ep.get('twr', 'N/A')})..."
        )

        result = record_clip(
            checkpoint_dir=args.checkpoint_dir,
            physics_config=physics_config,
            output_path=output_path,
            variant=variant,
            seed=episode_seed,
            corruption=args.corruption,
            corruption_sigma=args.corruption_sigma,
        )

        print(
            f"    -> {result['outcome']}, {result['n_steps']} steps, "
            f"reward={result['total_reward']:.1f}"
        )

    print(f"\nDone. {len(episodes)} clips saved to {output_dir}/")


if __name__ == "__main__":
    main()
