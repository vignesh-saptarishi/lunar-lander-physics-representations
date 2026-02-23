#!/usr/bin/env python
"""Run a batch of Lunar Lander RL training runs.

Reads a batch config YAML that lists training config names, then runs each
sequentially via train_rl.py --config. Each training config YAML defines
the full run (variant, algo, profile, run_dir, etc.).

Usage:
    # Standard 6-run matrix (3 variants x 2 algos)
    python lunar_lander/scripts/train_all_agents.py --batch all-agents

    # Just PPO variants
    python lunar_lander/scripts/train_all_agents.py --batch all-ppo

    # Show plan without running
    python lunar_lander/scripts/train_all_agents.py --batch full-matrix --dry-run

    # Resume interrupted batch
    python lunar_lander/scripts/train_all_agents.py --batch all-agents --resume

    # Multi-seed: run each config with 5 seeds, auto-creating unique run_dirs
    python lunar_lander/scripts/train_all_agents.py \\
        --batch multi-seed-full-variation \\
        --seeds 42,123,456,789,1024 \\
        --output-base ./data/networks
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.training_config import load_batch_config, load_training_config


def _build_run_matrix(runs, seeds, output_base):
    """Expand configs × seeds into (config_name, seed, run_dir) triples.

    When --seeds is provided, each config gets run once per seed. The run_dir
    is auto-constructed from --output-base + config_name + seed suffix:
        {output_base}/{config_name}/s{seed}

    Without --seeds, returns the original list (one run per config, seed=None).
    """
    if not seeds:
        return [(name, None, None) for name in runs]

    matrix = []
    for config_name in runs:
        for seed in seeds:
            # Auto-construct unique run_dir: output_base/config_name/s{seed}
            run_dir = str(Path(output_base) / config_name / f"s{seed}")
            matrix.append((config_name, seed, run_dir))
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Run a batch of Lunar Lander RL training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch config YAML (builtin name or file path). "
        "Contains a flat list of training config names.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Pass --resume to each training run"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show plan without running"
    )

    # Multi-seed support: run each config with multiple seeds
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. '42,123,456,789,1024'). "
        "Each config runs once per seed with unique run_dir.",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base directory for multi-seed run_dirs. Required "
        "when --seeds is provided. Run dirs are auto-created "
        "as: {output-base}/{config-name}/s{seed}",
    )

    args = parser.parse_args()

    # Parse seeds
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        if not args.output_base:
            parser.error("--output-base is required when --seeds is provided")

    # Load batch config
    runs = load_batch_config(args.batch)

    # Validate all training configs exist before starting.
    # Fail fast rather than discovering a missing config mid-batch.
    for config_name in runs:
        try:
            load_training_config(config_name)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # Build the full run matrix (configs × seeds)
    matrix = _build_run_matrix(runs, seeds, args.output_base)

    # Print matrix
    print("=" * 60)
    if seeds:
        print(
            f"Batch: {args.batch} — {len(runs)} configs × {len(seeds)} seeds "
            f"= {len(matrix)} runs"
        )
        print(f"Seeds: {seeds}")
        print(f"Output base: {args.output_base}")
    else:
        print(f"Batch: {args.batch} — {len(matrix)} runs")
    print("=" * 60)

    for i, (config_name, seed, run_dir) in enumerate(matrix):
        config = load_training_config(config_name)
        net_info = f" net={config['net_arch']}" if config["net_arch"] else ""
        seed_info = f" seed={seed}" if seed is not None else ""
        print(
            f"  {i+1}. {config_name:<40s} "
            f"{config['total_steps']:>10,} steps{net_info}{seed_info}"
        )
        if run_dir:
            print(f"      → {run_dir}")
    print()

    if args.dry_run:
        print("(dry run)")
        return

    # Execute
    completed = 0
    failed = []
    batch_start = time.time()

    skipped = 0
    for i, (config_name, seed, run_dir) in enumerate(matrix):
        seed_label = f" (seed={seed})" if seed is not None else ""
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(matrix)}] Starting: {config_name}{seed_label}")
        print(f"{'='*60}")

        # Skip runs that already have a final model.zip (fully trained).
        if args.resume and run_dir is not None:
            model_path = os.path.join(run_dir, "model.zip")
            if os.path.exists(model_path):
                print(f"  Already complete (model.zip exists). Skipping.")
                skipped += 1
                completed += 1
                continue

        cmd = [
            sys.executable,
            "lunar_lander/scripts/train_rl.py",
            "--config",
            config_name,
        ]
        # Override seed and run_dir for multi-seed mode
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if run_dir is not None:
            cmd.extend(["--run-dir", run_dir])
        if args.resume:
            cmd.append("--resume")

        run_start = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        run_elapsed = time.time() - run_start

        run_label = f"{config_name}{seed_label}"
        if result.returncode == 0:
            completed += 1
            print(f"  Completed in {run_elapsed:.0f}s ({run_elapsed/60:.1f}min)")
        else:
            failed.append(run_label)
            print(f"  FAILED (exit {result.returncode}) after {run_elapsed:.0f}s")

    batch_elapsed = time.time() - batch_start
    print(f"\n{'='*60}")
    print(
        f"Batch complete: {completed}/{len(matrix)} succeeded"
        f"{f' ({skipped} skipped)' if skipped else ''}"
    )
    if failed:
        print(f"  Failed:")
        for f in failed:
            print(f"    - {f}")
    print(f"  Wall time: {batch_elapsed:.0f}s ({batch_elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
