#!/usr/bin/env python
"""Behavioral analysis for a single trained Lunar Lander agent.

Chains the full analysis pipeline:
  1. Load .npz trajectory files (from prior collect_trajectories.py run)
  2. Compute per-episode action histograms (behavioral_metrics)
  3. Load or compute metrics.csv (trajectory_metrics, with new scalars)
  4. Aggregate to model-level distributions (behavioral_comparison)
  5. Compute adaptation score and binned landed% vs TWR
  6. Generate 3 plots + summary JSON

Requires collected trajectories to already exist in the agent's
trajectories/ directory. Run collect_trajectories.py first if needed.

Usage:
    # Analyze a single agent
    python lunar_lander/scripts/analyze_behavior.py \
        --agent-dir /path/to/trained/agent

    # Custom output directory
    python lunar_lander/scripts/analyze_behavior.py \
        --agent-dir /path/to/agent --output-dir /tmp/analysis

    # Adjust histogram resolution
    python lunar_lander/scripts/analyze_behavior.py \
        --agent-dir /path/to/agent --n-bins 100
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to path for imports.
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lunar_lander.src.analysis.behavioral_metrics import (
    compute_collection_histograms,
)
from lunar_lander.src.analysis.behavioral_comparison import (
    aggregate_model_distribution,
    compute_adaptation_score,
    compute_binned_performance,
)
from lunar_lander.src.analysis.trajectory_metrics import (
    compute_collection_metrics,
)
from lunar_lander.src.physics_config import LunarLanderPhysicsConfig


def _find_trajectories_dir(agent_dir: Path) -> Path:
    """Find the trajectories directory for an agent.

    Looks for a trajectories/ subdirectory containing .npz files.
    """
    traj_dir = agent_dir / "trajectories"
    if traj_dir.exists() and list(traj_dir.glob("*.npz")):
        return traj_dir
    # Maybe the agent_dir itself contains .npz files
    if list(agent_dir.glob("*.npz")):
        return agent_dir
    raise FileNotFoundError(
        f"No .npz trajectory files found in {agent_dir}/trajectories/ "
        f"or {agent_dir}/. Run collect_trajectories.py first."
    )


def _plot_action_distributions(model_dist: dict, output_path: Path):
    """Plot 1: Two-panel action distribution histograms.

    Left panel: P(main_thrust) over [0, 1].
    Right panel: P(side_thrust) over [-1, 1].

    These reveal the agent's control strategy — bang-bang (peaks at 0
    and 1) vs proportional (spread across range) vs biased (asymmetric).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Main thrust
    edges = model_dist["main_thrust_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax1.bar(
        centers,
        model_dist["main_thrust_probs"],
        width=edges[1] - edges[0],
        alpha=0.8,
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    ax1.set_xlabel("Main Thrust")
    ax1.set_ylabel("Probability")
    ax1.set_title("P(main_thrust)")
    ax1.set_xlim(0, 1)

    # Side thrust
    edges = model_dist["side_thrust_edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ax2.bar(
        centers,
        model_dist["side_thrust_probs"],
        width=edges[1] - edges[0],
        alpha=0.8,
        color="coral",
        edgecolor="white",
        linewidth=0.3,
    )
    ax2.set_xlabel("Side Thrust")
    ax2.set_ylabel("Probability")
    ax2.set_title("P(side_thrust)")
    ax2.set_xlim(-1, 1)

    fig.suptitle("Model-Level Action Distributions (all episodes)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_adaptation_by_twr(adaptation_result: dict, output_path: Path):
    """Plot 2: Overlaid P(main_thrust) per TWR quartile.

    If the colored distributions shift across quartiles, the agent
    adapts its thrust strategy to different physics. If they overlap
    perfectly, it's one strategy regardless.
    """
    quartile_dists = adaptation_result["quartile_distributions"]
    boundaries = adaptation_result["quartile_boundaries"]
    score = adaptation_result["adaptation_score"]
    counts = adaptation_result["quartile_episode_counts"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(quartile_dists)))

    for i, (q_dist, (lo, hi), count, color) in enumerate(
        zip(quartile_dists, boundaries, counts, colors)
    ):
        edges = q_dist.get(
            "main_thrust_edges",
            np.linspace(0, 1, len(q_dist["main_thrust_probs"]) + 1),
        )
        centers = (edges[:-1] + edges[1:]) / 2
        label = f"Q{i+1}: TWR [{lo:.1f}, {hi:.1f}] (n={count})"
        ax.step(
            centers,
            q_dist["main_thrust_probs"],
            where="mid",
            alpha=0.8,
            color=color,
            label=label,
            linewidth=1.5,
        )

    ax.set_xlabel("Main Thrust")
    ax.set_ylabel("Probability")
    ax.set_title(f"P(main_thrust) by TWR Quartile — Adaptation Score: {score:.4f}")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_landed_vs_twr(binned_perf: pd.DataFrame, output_path: Path):
    """Plot 3: Bar chart of landed% per TWR bin.

    Shows the agent's capability boundary — at what physics regime
    does it start failing?
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = range(len(binned_perf))
    bars = ax.bar(
        x,
        binned_perf["landed_pct"],
        color="seagreen",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate with episode counts
    for bar, row in zip(bars, binned_perf.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"n={row.n_episodes}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        binned_perf["bin_label"],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("TWR Bin")
    ax.set_ylabel("Landed %")
    ax.set_title("Landing Success Rate by TWR")
    ax.set_ylim(0, 110)  # Room for annotations

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# Profile difficulty bands for each parameter. These define the sampling
# ranges that easy/medium/hard profiles use, read from the YAML files.
# Params not overridden by a profile use full RANGES at all difficulties.
# Format: {param: {"easy": (lo, hi), "medium": (lo, hi), "hard": (lo, hi)}}
# Scalar overrides (fixed values) are stored as (val, val).
_DIFFICULTY_BANDS = {
    # Vehicle params — profiles don't constrain these, all use full RANGES
    "main_engine_power": {
        "easy": (5.0, 25.0),
        "medium": (5.0, 25.0),
        "hard": (5.0, 25.0),
    },
    "side_engine_power": {
        "easy": (0.2, 1.5),
        "medium": (0.2, 1.5),
        "hard": (0.2, 1.5),
    },
    "lander_density": {
        "easy": (2.5, 10.0),
        "medium": (2.5, 10.0),
        "hard": (2.5, 10.0),
    },
    "angular_damping": {
        "easy": (2.0, 5.0),  # high damping = stable
        "medium": (0.0, 3.0),  # moderate
        "hard": (0.0, 2.0),  # low damping = wobbly
    },
    # Physics params
    "gravity": {  # profiles don't constrain, full range
        "easy": (-12.0, -2.0),
        "medium": (-12.0, -2.0),
        "hard": (-12.0, -2.0),
    },
    "wind_power": {
        "easy": (0.0, 0.0),  # no wind
        "medium": (0.0, 15.0),  # moderate wind
        "hard": (0.0, 30.0),  # full wind
    },
    "turbulence_power": {
        "easy": (0.0, 0.0),  # no turbulence
        "medium": (0.0, 2.0),  # moderate
        "hard": (0.0, 5.0),  # full range
    },
}

# Split into vehicle (body) and physics (world) groups.
_VEHICLE_PARAMS = [
    "main_engine_power",
    "side_engine_power",
    "lander_density",
    "angular_damping",
]
_PHYSICS_PARAMS = ["gravity", "wind_power", "turbulence_power"]

# Band colors: easy=green, medium=orange, hard=red, with low alpha.
_BAND_COLORS = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}


def _plot_physics_distribution(episodes_df: pd.DataFrame, output_path: Path):
    """Plot 4: Physics parameter distributions vs difficulty bands.

    Two subplots: vehicle params (left) and physics params (right).
    Each param is a box plot showing the actual episode distribution,
    normalized to [0,1] relative to its full RANGES. Colored horizontal
    bands show where easy/medium/hard profiles sample from.
    """
    ranges = LunarLanderPhysicsConfig.RANGES

    def _normalize(values, param_name):
        """Normalize values to [0,1] relative to full RANGES."""
        lo, hi = ranges[param_name]
        return (np.asarray(values) - lo) / (hi - lo)

    def _plot_group(ax, param_names, title):
        """Draw box plots + difficulty bands for a group of params."""
        box_data = []
        labels = []
        for param in param_names:
            if param in episodes_df.columns:
                vals = episodes_df[param].dropna().values
                box_data.append(_normalize(vals, param))
            else:
                box_data.append(np.array([]))
            # Short display name: drop underscores, abbreviate
            labels.append(param.replace("_", "\n"))

        positions = list(range(len(param_names)))

        # Draw difficulty bands first (behind box plots).
        # For params where all bands are identical (full range at every
        # difficulty), draw a single gray band instead of three overlapping
        # colored bands that blend into a muddy tint.
        for param_idx, param in enumerate(param_names):
            bands = _DIFFICULTY_BANDS.get(param, {})
            if not bands:
                continue

            full_lo, full_hi = ranges[param]
            # Normalize all band bounds
            norm_bands = {}
            for diff, (lo, hi) in bands.items():
                norm_bands[diff] = (
                    (lo - full_lo) / (full_hi - full_lo),
                    (hi - full_lo) / (full_hi - full_lo),
                )

            # Check if all bands are identical (unconstrained param)
            unique_bounds = set(norm_bands.values())
            if len(unique_bounds) == 1:
                # Profile allows full range at all difficulties, but TWR
                # rejection sampling still shapes the effective distribution.
                nlo, nhi = unique_bounds.pop()
                ax.fill_between(
                    [param_idx - 0.4, param_idx + 0.4],
                    nlo,
                    nhi,
                    alpha=0.08,
                    color="gray",
                    linewidth=0,
                )
                ax.text(
                    param_idx,
                    0.5,
                    "TWR\nfiltered",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="silver",
                    fontstyle="italic",
                )
            else:
                # Draw from widest (hard) to narrowest (easy) so
                # narrower bands layer on top of wider ones.
                for difficulty in ["hard", "medium", "easy"]:
                    if difficulty not in norm_bands:
                        continue
                    nlo, nhi = norm_bands[difficulty]
                    color = _BAND_COLORS[difficulty]
                    ax.fill_between(
                        [param_idx - 0.4, param_idx + 0.4],
                        nlo,
                        nhi,
                        alpha=0.18,
                        color=color,
                        linewidth=0,
                    )

        # Box plots on top.
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Normalized value (0=min, 1=max)")
        ax.set_ylim(-0.12, 1.12)
        ax.set_title(title)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.axhline(1, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

        # Annotate actual min/max values from RANGES below and above each box.
        # For params with no profile overrides, label "full range" so the
        # absence of colored bands is clearly intentional.
        for param_idx, param in enumerate(param_names):
            lo, hi = ranges[param]
            lo_str = f"{lo:g}"
            hi_str = f"{hi:g}"
            ax.text(
                param_idx,
                -0.08,
                lo_str,
                ha="center",
                va="top",
                fontsize=7,
                color="gray",
            )
            ax.text(
                param_idx,
                1.08,
                hi_str,
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    _plot_group(ax1, _VEHICLE_PARAMS, "Vehicle (Body) Parameters")
    _plot_group(ax2, _PHYSICS_PARAMS, "Physics (World) Parameters")

    # Shared legend for difficulty bands.
    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=_BAND_COLORS["easy"], alpha=0.3, label="easy"),
        Patch(facecolor=_BAND_COLORS["medium"], alpha=0.3, label="medium"),
        Patch(facecolor=_BAND_COLORS["hard"], alpha=0.3, label="hard"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    n_eps = len(episodes_df)
    fig.suptitle(
        f"Physics Parameter Distributions (n={n_eps} episodes)",
        fontsize=13,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral analysis for a single trained Lunar Lander agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent-dir",
        required=True,
        type=str,
        help="Path to trained agent checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save plots and summary "
        "(default: agent_dir/behavioral_analysis/)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    parser.add_argument(
        "--n-quartiles",
        type=int,
        default=4,
        help="Number of quartiles for adaptation score (default: 4)",
    )
    parser.add_argument(
        "--twr-bins",
        type=int,
        default=5,
        help="Number of bins for landed%% vs TWR (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for histogram computation (default: 8)",
    )
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir else agent_dir / "behavioral_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_name = agent_dir.name
    print(f"\n=== Behavioral Analysis: {agent_name} ===\n")

    # Step 1: Find trajectories
    traj_dir = _find_trajectories_dir(agent_dir)
    npz_files = sorted(traj_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} trajectory files in {traj_dir}")

    # Step 2: Compute per-episode action histograms
    print(
        f"\nComputing action histograms "
        f"(n_bins={args.n_bins}, workers={args.workers})..."
    )
    histograms = compute_collection_histograms(
        str(traj_dir),
        workers=args.workers,
        n_bins=args.n_bins,
    )
    print(f"  Computed histograms for {len(histograms)} episodes")

    # Step 3: Load or compute metrics.csv
    metrics_path = traj_dir / "metrics.csv"
    if metrics_path.exists():
        print(f"\nLoading existing metrics from {metrics_path}")
        episodes_df = pd.read_csv(str(metrics_path))
    else:
        print(f"\nComputing episode metrics (workers={args.workers})...")
        episodes_df = compute_collection_metrics(
            str(traj_dir),
            workers=args.workers,
        )
        episodes_df.to_csv(str(metrics_path), index=False)
        print(f"  Saved metrics to {metrics_path}")

    n_landed = (episodes_df["outcome"] == "landed").sum()
    landed_pct = (episodes_df["outcome"] == "landed").mean() * 100
    print(f"  {len(episodes_df)} episodes, {n_landed} landed ({landed_pct:.1f}%)")

    # Step 4: Aggregate to model-level distributions
    print("\nAggregating to model-level distributions...")
    model_dist = aggregate_model_distribution(list(histograms.values()))

    # Step 5: Compute adaptation score
    print(f"\nComputing adaptation score (n_quartiles={args.n_quartiles})...")
    adaptation_result = compute_adaptation_score(
        histograms,
        episodes_df,
        physics_col="twr",
        n_quartiles=args.n_quartiles,
    )
    print(f"  Adaptation score: {adaptation_result['adaptation_score']:.4f}")
    for i, ((lo, hi), count) in enumerate(
        zip(
            adaptation_result["quartile_boundaries"],
            adaptation_result["quartile_episode_counts"],
        )
    ):
        print(f"  Q{i+1}: TWR [{lo:.2f}, {hi:.2f}] — {count} episodes")

    # Step 6: Compute binned performance
    print(f"\nComputing binned performance (n_bins={args.twr_bins})...")
    binned_perf = compute_binned_performance(
        episodes_df,
        bin_col="twr",
        n_bins=args.twr_bins,
    )
    for _, row in binned_perf.iterrows():
        print(
            f"  TWR {row['bin_label']}: {row['landed_pct']:.1f}% landed "
            f"(n={row['n_episodes']}, mean_reward={row['mean_reward']:.1f})"
        )

    # Step 7: Generate plots
    print(f"\nGenerating plots in {output_dir}...")
    _plot_action_distributions(
        model_dist,
        output_dir / "action_distributions.png",
    )
    _plot_adaptation_by_twr(
        adaptation_result,
        output_dir / "adaptation_by_twr.png",
    )
    _plot_landed_vs_twr(binned_perf, output_dir / "landed_vs_twr.png")
    _plot_physics_distribution(episodes_df, output_dir / "physics_distribution.png")

    # Step 8: Save summary JSON
    summary = {
        "agent_name": agent_name,
        "n_episodes": len(episodes_df),
        "landed_pct": float(landed_pct),
        "adaptation_score": adaptation_result["adaptation_score"],
        "quartile_boundaries": adaptation_result["quartile_boundaries"],
        "quartile_episode_counts": adaptation_result["quartile_episode_counts"],
        "binned_performance": binned_perf.to_dict(orient="records"),
    }
    # Add new scalar summaries if available
    new_scalars = [
        "std_main_thrust",
        "std_side_thrust",
        "main_thrust_frac_full",
        "main_thrust_frac_zero",
        "frac_descending",
        "frac_hovering",
        "frac_approaching",
        "frac_correcting",
        "thrust_autocorr_lag1",
        "side_thrust_autocorr_lag1",
    ]
    for col in new_scalars:
        if col in episodes_df.columns:
            summary[f"mean_{col}"] = float(episodes_df[col].mean())

    summary_path = output_dir / "behavioral_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n=== Done. Outputs in {output_dir} ===\n")


if __name__ == "__main__":
    main()
