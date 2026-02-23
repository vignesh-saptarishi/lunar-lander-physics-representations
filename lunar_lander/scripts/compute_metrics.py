#!/usr/bin/env python
"""Compute per-episode metrics from collected Lunar Lander trajectories.

Reads .npz trajectory files (produced by collect_trajectories.py) and
outputs a metrics.csv with one row per episode and ~20 scalar columns.
Also prints a summary table to stdout.

For the full pipeline (collect + metrics), use run_eval_pipeline.py.

Usage:
    # Single collection directory
    python lunar_lander/scripts/compute_metrics.py /path/to/trajectories/

    # Custom output path
    python lunar_lander/scripts/compute_metrics.py /path/to/trajectories/ \
        --output /tmp/my_metrics.csv

    # Control parallelism
    python lunar_lander/scripts/compute_metrics.py /path/to/trajectories/ \
        --workers 4
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.analysis.trajectory_metrics import compute_collection_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-episode metrics from collected trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "dir", type=str, help="Directory containing .npz trajectory files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: {dir}/metrics.csv)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel workers (default: 8)"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: {args.dir} is not a directory")
        sys.exit(1)

    npz_count = len(list(Path(args.dir).glob("*.npz")))
    if npz_count == 0:
        print(f"ERROR: No .npz files found in {args.dir}")
        sys.exit(1)

    print(f"Computing metrics for {npz_count} episodes in {args.dir}/")
    print(f"  Workers: {args.workers}")

    df = compute_collection_metrics(args.dir, workers=args.workers)

    # Save CSV
    csv_path = args.output or os.path.join(args.dir, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    n_total = len(df)
    n_landed = int((df["outcome"] == "landed").sum())
    n_crashed = int((df["outcome"] == "crashed").sum())
    n_timeout = int((df["outcome"] == "timeout").sum())
    n_oob = int((df["outcome"] == "out_of_bounds").sum())

    print(f"  Episodes:   {n_total}")
    print(f"  Landed:     {n_landed}/{n_total} ({100*n_landed/n_total:.0f}%)")
    print(f"  Crashed:    {n_crashed}/{n_total} ({100*n_crashed/n_total:.0f}%)")
    if n_oob > 0:
        print(f"  OOB:        {n_oob}/{n_total} ({100*n_oob/n_total:.0f}%)")
    if n_timeout > 0:
        print(f"  Timeout:    {n_timeout}/{n_total} ({100*n_timeout/n_total:.0f}%)")

    print(
        f"\n  Mean reward:        {df['total_reward'].mean():>8.1f} "
        f"+/- {df['total_reward'].std():.1f}"
    )
    print(f"  Mean steps:         {df['episode_steps'].mean():>8.1f}")
    print(f"  Mean thrust duty:   {df['thrust_duty_cycle'].mean():>8.2f}")
    print(f"  Mean total fuel:    {df['total_fuel'].mean():>8.1f}")

    # Landed-only metrics
    landed_df = df[df["outcome"] == "landed"]
    if len(landed_df) > 0:
        print(f"\n  Landed episodes:")
        print(f"    Mean landing x error:  {landed_df['landing_x_error'].mean():.3f}")
        print(f"    Mean landing vy:       {landed_df['landing_vy'].mean():.3f}")
        print(f"    Mean fuel efficiency:  {landed_df['fuel_efficiency'].mean():.3f}")
        print(
            f"    Mean angle at landing: {landed_df['angle_at_landing'].mean():.4f} rad"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
