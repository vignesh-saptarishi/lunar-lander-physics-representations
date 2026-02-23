#!/usr/bin/env python3
"""Aggregate multi-seed training runs for one experiment.

Reads a seed-aggregation manifest, parses TensorBoard events from all
seeds of each config, computes cross-seed statistics (mean/std/CV),
and writes per-config JSON artifacts + plots + experiment summary table.

Usage:
    # All configs in a manifest
    python aggregate_seeds.py --manifest seed-agg/parametric-vs-behavioral

    # Single config (quick check)
    python aggregate_seeds.py --manifest seed-agg/parametric-vs-behavioral \
        --config full-variation/blind-ppo-easy-128-lowent

    # Dry run
    python aggregate_seeds.py --manifest seed-agg/parametric-vs-behavioral --dry-run

Design spec: analysis-tooling.md Sections 4-5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.analysis.manifest import load_analysis_manifest
from lunar_lander.src.analysis.seed_aggregation import (
    aggregate_seed_metrics,
    plot_learning_curves,
    plot_summary,
    write_config_outputs,
    write_experiment_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest name (e.g., 'seed-agg/parametric-vs-behavioral') or file path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Process only this config (e.g., 'full-variation/blind-ppo-easy-128-lowent').",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Override output location (default: from manifest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if outputs already exist.",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=5,
        help="Number of final eval checkpoints to average (default: 5).",
    )
    args = parser.parse_args()

    # --- Load manifest ---
    manifest = load_analysis_manifest(args.manifest)
    output_base = args.output_base or manifest.get("output_base", ".")
    experiment_name = manifest["experiment"]

    print(f"Experiment: {experiment_name}")
    print(f"Output:     {output_base}")

    # --- Filter configs ---
    configs = manifest["configs"]
    if args.config:
        if args.config not in configs:
            print(
                f"ERROR: Config '{args.config}' not in manifest. "
                f"Available: {list(configs.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    print(f"Configs:    {len(configs)}")
    print()

    # --- Validate seed directories exist ---
    missing = []
    for config_name, config_data in configs.items():
        for seed_dir in config_data.get("seed_dirs", []):
            if not Path(seed_dir).exists():
                missing.append(f"  {config_name}: {seed_dir}")
    if missing:
        print("ERROR: Missing seed directories:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)

    # --- Dry run ---
    if args.dry_run:
        print("DRY RUN -- would process:")
        for config_name, config_data in sorted(configs.items()):
            seeds = config_data.get("seeds", [])
            condition = config_data.get("condition", "?")
            print(f"  {config_name}  (N={len(seeds)}, condition={condition})")
            for sd in config_data.get("seed_dirs", []):
                print(f"    -> {sd}")
        return

    # --- Process each config ---
    all_results = {}
    for config_name, config_data in sorted(configs.items()):
        condition = config_data.get("condition", "")
        # Output goes to {output_base}/{condition}/{config_short_name}/.
        # Config name may include condition prefix (e.g., "full-variation/blind-ppo..."),
        # so we use condition from metadata to build the directory.
        config_short = config_name.split("/")[-1] if "/" in config_name else config_name
        config_output_dir = str(Path(output_base) / condition / config_short)

        # Skip if outputs exist (unless --force).
        metrics_path = Path(config_output_dir) / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"  SKIP {config_name} (outputs exist, use --force to overwrite)")
            # Still load existing metrics for the summary table.
            import json

            with open(metrics_path) as f:
                existing = json.load(f)
            all_results[config_name] = {
                "n_seeds": existing.get("n_seeds"),
                "training_metrics": existing.get("training_metrics", {}),
            }
            continue

        seeds = config_data.get("seeds", [])
        seed_dirs = config_data.get("seed_dirs", [])
        print(f"  Processing {config_name} (N={len(seeds)})...")

        # Aggregate.
        agg_result = aggregate_seed_metrics(
            seed_dirs=seed_dirs,
            seeds=seeds,
            last_k=args.last_k,
        )

        # Write JSON artifacts.
        write_config_outputs(
            config_name=config_short,
            config_data=config_data,
            agg_result=agg_result,
            output_dir=config_output_dir,
        )

        # Plot learning curves.
        plot_learning_curves(
            learning_curves=agg_result["learning_curves"],
            seeds=seeds,
            config_name=config_short,
            output_path=str(Path(config_output_dir) / "learning_curves.png"),
        )

        # Plot summary (eval metrics + training dynamics).
        plot_summary(
            training_metrics=agg_result["training_metrics"],
            seeds=seeds,
            config_name=config_short,
            output_path=str(Path(config_output_dir) / "summary.png"),
            training_dynamics=agg_result.get("training_dynamics"),
        )

        # Track for experiment summary.
        all_results[config_name] = {
            "n_seeds": len(seeds),
            "training_metrics": agg_result["training_metrics"],
        }

        # Print per-config result.
        landed = agg_result["training_metrics"].get("landed_pct", {})
        reward = agg_result["training_metrics"].get("mean_reward", {})
        consistent = agg_result["seed_consistency"].get("consistent", "?")
        print(
            f"    landed={landed.get('mean', '?')} +/- {landed.get('std', '?')}  "
            f"reward={reward.get('mean', '?')} +/- {reward.get('std', '?')}  "
            f"consistent={consistent}"
        )

    # --- Write experiment summary ---
    if all_results:
        write_experiment_summary(
            experiment_name=experiment_name,
            configs_results=all_results,
            output_dir=output_base,
        )

        # Print summary table to stdout.
        print()
        summary_path = Path(output_base) / "summary_table.txt"
        if summary_path.exists():
            print(summary_path.read_text())


if __name__ == "__main__":
    main()
