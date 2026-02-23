"""Model-level behavioral aggregation and adaptation scoring.

Operates on collections of per-episode histograms and metrics to
produce Level 3 analysis outputs:

    aggregate_model_distribution(histograms) -> dict
        Sum per-episode histograms into a model-level probability distribution.

    compute_adaptation_score(histograms, df, physics_col, n_quartiles) -> dict
        Partition episodes by physics quartile, compute per-quartile
        distributions, measure JS divergence between all pairs.
        This is THE central metric: does the agent change its behavior
        when the physics changes?

    compute_binned_performance(df, bin_col, n_bins) -> DataFrame
        Bin episodes by a physics parameter and compute landed% per bin.
        Shows the agent's capability boundary across the physics spectrum.

The adaptation score uses Jensen-Shannon divergence (symmetric, bounded
[0, 1], well-defined for discrete distributions). scipy.spatial.distance
.jensenshannon returns the JSD *distance* (square root of the divergence),
which is already in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def aggregate_model_distribution(histograms: list[dict]) -> dict:
    """Aggregate per-episode histograms into a model-level distribution.

    Sums the raw bin counts across all episodes, then normalizes to
    a probability distribution. This treats every timestep equally —
    longer episodes contribute more counts, which is correct because
    longer episodes have more actions.

    Args:
        histograms: List of dicts from compute_action_histograms().
            Each must have 'main_thrust_counts', 'main_thrust_edges',
            'side_thrust_counts', 'side_thrust_edges'.

    Returns:
        Dict with keys:
            main_thrust_probs: (n_bins,) float — probability distribution
            main_thrust_edges: (n_bins+1,) float — bin edges
            side_thrust_probs: (n_bins,) float — probability distribution
            side_thrust_edges: (n_bins+1,) float — bin edges

    Raises:
        ValueError: If histograms list is empty.
    """
    if not histograms:
        raise ValueError("Cannot aggregate empty histogram list")

    # Sum counts across all episodes.
    main_total = np.sum([h["main_thrust_counts"] for h in histograms], axis=0).astype(
        float
    )
    side_total = np.sum([h["side_thrust_counts"] for h in histograms], axis=0).astype(
        float
    )

    # Normalize to probability distributions.
    main_sum = main_total.sum()
    side_sum = side_total.sum()

    main_probs = main_total / main_sum if main_sum > 0 else main_total
    side_probs = side_total / side_sum if side_sum > 0 else side_total

    return {
        "main_thrust_probs": main_probs,
        "main_thrust_edges": histograms[0]["main_thrust_edges"],
        "side_thrust_probs": side_probs,
        "side_thrust_edges": histograms[0]["side_thrust_edges"],
    }


def compute_adaptation_score(
    histograms: dict[str, dict],
    episodes_df: pd.DataFrame,
    physics_col: str = "twr",
    n_quartiles: int = 4,
) -> dict:
    """Compute adaptation score: how much does the action distribution
    change across physics quartiles?

    Partitions episodes into quartiles by the specified physics column,
    computes per-quartile aggregate distributions, then measures the
    mean pairwise Jensen-Shannon divergence between all quartile pairs.

    A high score means the agent uses different strategies for different
    physics regimes (it adapts). A low score means one-size-fits-all.

    Args:
        histograms: Dict mapping npz_path -> histogram dict (from
            compute_collection_histograms).
        episodes_df: DataFrame with at least 'npz_path' and physics_col.
            The npz_path column must match the keys in histograms — this
            is used to join physics info to histograms.
        physics_col: Column to partition by (default: 'twr').
        n_quartiles: Number of quantile bins (default: 4).

    Returns:
        Dict with keys:
            adaptation_score: float in [0, 1] — mean pairwise JS distance
            pairwise_js: (n_quartiles, n_quartiles) float array — JS matrix
            quartile_boundaries: list of (low, high) tuples
            n_quartiles: int — actual number of quartiles used
            quartile_episode_counts: list of int — episodes per quartile
            quartile_distributions: list of dicts — per-quartile distributions
    """
    # Match histogram keys to DataFrame rows via npz_path.
    # Extract just the filename to handle path mismatches.
    df = episodes_df.copy()
    df["_npz_basename"] = df["npz_path"].apply(lambda p: str(p).rsplit("/", 1)[-1])

    hist_by_basename = {}
    for key, hist in histograms.items():
        basename = str(key).rsplit("/", 1)[-1]
        hist_by_basename[basename] = hist

    # Compute quartile boundaries.
    quantiles = np.linspace(0, 1, n_quartiles + 1)
    boundaries = np.quantile(df[physics_col].values, quantiles)

    # Assign each episode to a quartile.
    # duplicates="drop" handles conditions where the physics column has
    # near-zero variance (e.g. turbulence-only where TWR is constant).
    unique_boundaries = np.unique(boundaries)
    if len(unique_boundaries) < 2:
        # All values identical — no variation to measure adaptation against.
        val = float(unique_boundaries[0])
        return {
            "adaptation_score": 0.0,
            "pairwise_js": np.zeros((1, 1)),
            "quartile_boundaries": [(val, val)],
            "n_quartiles": 1,
            "quartile_episode_counts": [len(df)],
            "quartile_distributions": [],
        }
    else:
        df["_quartile"] = pd.cut(
            df[physics_col],
            bins=unique_boundaries,
            labels=range(len(unique_boundaries) - 1),
            include_lowest=True,
            duplicates="drop",
        ).astype(int)
        n_quartiles = len(unique_boundaries) - 1

    # Compute per-quartile aggregate distributions.
    quartile_dists = []
    quartile_counts = []
    for q in range(n_quartiles):
        q_df = df[df["_quartile"] == q]
        q_hists = []
        for _, row in q_df.iterrows():
            basename = row["_npz_basename"]
            if basename in hist_by_basename:
                q_hists.append(hist_by_basename[basename])

        if q_hists:
            q_dist = aggregate_model_distribution(q_hists)
        else:
            # Empty quartile — uniform distribution as fallback.
            n_bins = len(next(iter(histograms.values()))["main_thrust_counts"])
            q_dist = {
                "main_thrust_probs": np.ones(n_bins) / n_bins,
                "side_thrust_probs": np.ones(n_bins) / n_bins,
            }

        quartile_dists.append(q_dist)
        quartile_counts.append(len(q_hists))

    # Compute pairwise JS divergence matrix (main thrust only for
    # the scalar adaptation score — side thrust is less informative
    # for physics adaptation since it's primarily used for stability).
    pairwise = np.zeros((n_quartiles, n_quartiles))
    for i in range(n_quartiles):
        for j in range(i + 1, n_quartiles):
            js = jensenshannon(
                quartile_dists[i]["main_thrust_probs"],
                quartile_dists[j]["main_thrust_probs"],
            )
            pairwise[i, j] = js
            pairwise[j, i] = js

    # Adaptation score = mean of upper triangle (all unique pairs).
    n_pairs = n_quartiles * (n_quartiles - 1) / 2
    adaptation_score = float(pairwise.sum() / (2 * n_pairs)) if n_pairs > 0 else 0.0

    # Build quartile boundary tuples for reporting.
    boundary_tuples = [
        (float(unique_boundaries[i]), float(unique_boundaries[i + 1]))
        for i in range(n_quartiles)
    ]

    return {
        "adaptation_score": adaptation_score,
        "pairwise_js": pairwise,
        "quartile_boundaries": boundary_tuples,
        "n_quartiles": n_quartiles,
        "quartile_episode_counts": quartile_counts,
        "quartile_distributions": quartile_dists,
    }


def compute_binned_performance(
    episodes_df: pd.DataFrame,
    bin_col: str = "twr",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Compute landed% and mean reward per physics bin.

    Bins episodes by a physics parameter and computes performance
    metrics per bin. This shows the agent's capability boundary —
    at what physics regime does it start failing?

    Args:
        episodes_df: DataFrame with at least bin_col, 'outcome', and
            'total_reward' columns.
        bin_col: Column to bin by (default: 'twr').
        n_bins: Number of equal-width bins.

    Returns:
        DataFrame with columns:
            bin_center: float — center of the bin
            bin_label: str — human-readable bin range
            n_episodes: int — number of episodes in this bin
            landed_pct: float — percentage of episodes that landed
            mean_reward: float — mean total reward in this bin
    """
    df = episodes_df.copy()

    # Create equal-width bins.
    df["_bin"], bin_edges = pd.cut(
        df[bin_col],
        bins=n_bins,
        retbins=True,
        include_lowest=True,
    )

    rows = []
    for interval in sorted(df["_bin"].dropna().unique()):
        bin_df = df[df["_bin"] == interval]
        n = len(bin_df)
        landed = (bin_df["outcome"] == "landed").sum()
        landed_pct = (landed / n * 100) if n > 0 else 0.0

        rows.append(
            {
                "bin_center": float(interval.mid),
                "bin_label": str(interval),
                "n_episodes": n,
                "landed_pct": float(landed_pct),
                "mean_reward": float(bin_df["total_reward"].mean()),
            }
        )

    return pd.DataFrame(rows)
