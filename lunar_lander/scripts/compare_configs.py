#!/usr/bin/env python3
"""Compare agent configs from multi-seed evaluation runs.

Reads a comparison manifest, loads episode metrics (metrics.csv) from
each seed, computes grouped statistics with Mann-Whitney U tests and
Cohen's d effect sizes, and writes tables, plots, and JSON outputs.

Usage:
    # All comparisons in a manifest
    python compare_configs.py --manifest comparison/parametric-vs-behavioral

    # Single comparison (quick check)
    python compare_configs.py --manifest comparison/parametric-vs-behavioral \
        --comparison full-variation-easy

    # Dry run — shows what would be processed without doing it
    python compare_configs.py --manifest comparison/parametric-vs-behavioral --dry-run

Design spec: analysis-tooling.md Section 6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.analysis.cross_config_comparison import (
    compute_variant_stats,
    load_behavioral_summaries,
    load_comparison_metrics,
    plot_behavioral_comparison,
    plot_outcome_breakdown,
    plot_performance_bars,
    run_statistical_tests,
    write_comparison_outputs,
)
from lunar_lander.src.analysis.manifest import load_comparison_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Compare agent configs from multi-seed evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "Manifest name (e.g., 'comparison/parametric-vs-behavioral') "
            "or file path."
        ),
    )
    parser.add_argument(
        "--comparison",
        default=None,
        help="Process only this comparison (e.g., 'full-variation-easy').",
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
    args = parser.parse_args()

    # --- Load manifest ---
    manifest = load_comparison_manifest(args.manifest)
    output_base = args.output_base or manifest.get("output_base", ".")
    experiment_name = manifest["experiment"]

    print(f"Experiment: {experiment_name}")
    print(f"Output:     {output_base}")

    # --- Filter comparisons ---
    comparisons = manifest["comparisons"]
    if args.comparison:
        if args.comparison not in comparisons:
            print(
                f"ERROR: Comparison '{args.comparison}' not in manifest. "
                f"Available: {list(comparisons.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        comparisons = {args.comparison: comparisons[args.comparison]}

    print(f"Comparisons: {len(comparisons)}")
    print()

    # --- Validate that metrics.csv exists for all seeds ---
    missing = []
    for comp_name, comp_data in comparisons.items():
        for variant, config in comp_data.get("configs", {}).items():
            traj_subdir = config.get("trajectory_subdir", "trajectories")
            for seed_dir in config.get("seed_dirs", []):
                csv_path = Path(seed_dir) / traj_subdir / "metrics.csv"
                if not csv_path.exists():
                    missing.append(f"  {comp_name}/{variant}: {csv_path}")
    if missing:
        print(
            "ERROR: Missing episode metrics (run eval pipeline first):",
            file=sys.stderr,
        )
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)

    # --- Dry run ---
    if args.dry_run:
        print("DRY RUN -- would process:")
        for comp_name, comp_data in sorted(comparisons.items()):
            variants = list(comp_data.get("configs", {}).keys())
            condition = comp_data.get("condition", "?")
            profile = comp_data.get("profile", "?")
            print(f"  {comp_name} ({condition}/{profile}): {variants}")
            for variant, config in comp_data.get("configs", {}).items():
                n_seeds = len(config.get("seeds", []))
                print(
                    f"    {variant}: N={n_seeds}, " f"seeds={config.get('seeds', [])}"
                )
        return

    # --- Process each comparison ---
    all_results = {}
    for comp_name, comp_data in sorted(comparisons.items()):
        # Skip if outputs exist (unless --force).
        stat_path = Path(output_base) / "stat_tests.json"
        if stat_path.exists() and not args.force and args.comparison is None:
            print("  SKIP experiment (stat_tests.json exists, use --force)")
            return

        print(f"  Processing {comp_name}...")

        # Load per-seed episode metrics.
        metrics_by_variant = load_comparison_metrics(comp_data)

        # Compute per-seed aggregate stats for each variant.
        # This is the core aggregation: 100 episodes/seed -> 1 value/seed.
        variant_stats = {}
        for variant, dfs in metrics_by_variant.items():
            variant_stats[variant] = compute_variant_stats(dfs)

        variant_names = list(comp_data.get("configs", {}).keys())

        # Run statistical tests (pairwise between first two variants).
        # With N=2 per group, Mann-Whitney U can't reach significance,
        # but we report the stats honestly.
        test_results = {}
        if len(variant_names) >= 2:
            test_results = run_statistical_tests(
                variant_stats,
                variant_names[:2],
            )

        # Load behavioral summaries (adaptation score etc.).
        behavioral = load_behavioral_summaries(comp_data)

        # Merge adaptation_score from behavioral summaries into variant_stats.
        for variant, summaries in behavioral.items():
            adaptation_scores = [
                s.get("adaptation_score", float("nan")) for s in summaries
            ]
            valid = [v for v in adaptation_scores if not np.isnan(v)]
            if valid and variant in variant_stats:
                variant_stats[variant]["adaptation_score"] = {
                    "mean": round(float(np.mean(valid)), 4),
                    "std": (
                        round(float(np.std(valid, ddof=1)), 4)
                        if len(valid) > 1
                        else 0.0
                    ),
                    "per_seed": [round(v, 4) for v in adaptation_scores],
                }

        # Collect results.
        all_results[comp_name] = {
            "condition": comp_data.get("condition"),
            "profile": comp_data.get("profile"),
            "variants": variant_names,
            "variant_stats": variant_stats,
            "test_results": test_results,
            "n_seeds": {
                v: len(comp_data["configs"][v].get("seeds", [])) for v in variant_names
            },
            "n_episodes": {
                v: sum(len(df) for df in metrics_by_variant[v]) for v in variant_names
            },
            "seed_dfs": metrics_by_variant,
            "seeds": {
                v: comp_data["configs"][v].get("seeds", []) for v in variant_names
            },
        }

        # Print per-comparison result.
        for variant in variant_names:
            vs = variant_stats.get(variant, {})
            landed = vs.get("landed_pct", {})
            reward = vs.get("mean_reward", {})
            print(
                f"    {variant}: "
                f"landed={landed.get('mean', '?')} +/- "
                f"{landed.get('std', '?')}  "
                f"reward={reward.get('mean', '?')} +/- "
                f"{reward.get('std', '?')}"
            )
        if test_results and "landed_pct" in test_results:
            tr = test_results["landed_pct"]
            print(
                f"    p={tr.get('p_value', '?')}  "
                f"d={tr.get('effect_size_cohens_d', '?')}  "
                f"winner={tr.get('winner', '?')}"
            )

        # Plot outcome breakdown per comparison.
        comp_output_dir = Path(output_base) / comp_name
        comp_output_dir.mkdir(parents=True, exist_ok=True)
        plot_outcome_breakdown(
            variant_stats,
            variant_names,
            comp_name,
            str(comp_output_dir / "outcome_breakdown.png"),
        )

    # --- Write experiment-level outputs ---
    if all_results:
        write_comparison_outputs(experiment_name, all_results, output_base)

        # Plot experiment-level charts.
        plot_performance_bars(
            all_results,
            str(Path(output_base) / "performance_bars.png"),
        )
        plot_behavioral_comparison(
            all_results,
            str(Path(output_base) / "behavioral_comparison.png"),
        )

        # Print comparison table to stdout.
        print()
        table_path = Path(output_base) / "comparison_table.txt"
        if table_path.exists():
            print(table_path.read_text())


if __name__ == "__main__":
    main()
