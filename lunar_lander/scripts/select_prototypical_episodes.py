#!/usr/bin/env python3
"""Select prototypical episodes from metrics.csv for article video clips.

Reads metrics.csv, applies behavioral filters, and outputs a JSON manifest
of selected episode paths + metadata. This manifest feeds into record_clips.py.

Usage:
    # Select 4 high-autocorrelation landed labeled episodes
    python lunar_lander/scripts/select_prototypical_episodes.py \
        --metrics /path/to/trajectories/metrics.csv \
        --n 4 --outcome landed \
        --sort-by thrust_autocorr_lag1 --descending \
        --filter "thrust_autocorr_lag1>=0.95" \
        --diversity-on gravity \
        --output selected.json

    # Select 4 low-autocorrelation landed blind episodes
    python lunar_lander/scripts/select_prototypical_episodes.py \
        --metrics /path/to/trajectories/metrics.csv \
        --n 4 --outcome landed \
        --sort-by thrust_autocorr_lag1 --ascending \
        --filter "thrust_autocorr_lag1<=0.75" \
        --output selected.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.clip_recording import select_episodes


def _parse_filter(filter_str: str) -> tuple[str, tuple[float | None, float | None]]:
    """Parse 'column>=0.5' or 'column<=0.8' into (column, (min, max))."""
    for op in [">=", "<="]:
        if op in filter_str:
            col, val = filter_str.split(op)
            val = float(val)
            if op == ">=":
                return col.strip(), (val, None)
            else:
                return col.strip(), (None, val)
    raise ValueError(
        f"Cannot parse filter: {filter_str}. Use column>=val or column<=val."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select prototypical episodes for article video clips.",
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--n", type=int, default=4, help="Number of episodes to select")
    parser.add_argument("--outcome", default="landed", help="Required outcome")
    parser.add_argument("--sort-by", required=True, help="Column to sort by")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending")
    parser.add_argument(
        "--descending", action="store_true", help="Sort descending (default)"
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        default=[],
        help="Filter expression (e.g. 'thrust_autocorr_lag1>=0.95'). Repeatable.",
    )
    parser.add_argument("--diversity-on", default=None, help="Column to diversify on")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--variant", default="", help="Variant label for metadata")
    parser.add_argument("--condition", default="", help="Condition label for metadata")

    args = parser.parse_args()

    ascending = args.ascending and not args.descending

    df = pd.read_csv(args.metrics)
    print(f"Loaded {len(df)} episodes from {args.metrics}")

    filters = {}
    for f in args.filters:
        col, bounds = _parse_filter(f)
        filters[col] = bounds

    results = select_episodes(
        df,
        n=args.n,
        outcome=args.outcome,
        sort_by=args.sort_by,
        ascending=ascending,
        filters=filters or None,
        diversity_on=args.diversity_on,
    )

    print(f"Selected {len(results)} episodes:")
    for r in results:
        print(
            f"  {Path(r['npz_path']).name}: {r['outcome']}, "
            f"{args.sort_by}={r[args.sort_by]:.3f}, "
            f"gravity={r.get('gravity', 'N/A'):.1f}"
        )

    output = {
        "selection_criteria": {
            "metrics_csv": args.metrics,
            "n": args.n,
            "outcome": args.outcome,
            "sort_by": args.sort_by,
            "ascending": ascending,
            "filters": args.filters,
            "diversity_on": args.diversity_on,
        },
        "variant": args.variant,
        "condition": args.condition,
        "episodes": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
