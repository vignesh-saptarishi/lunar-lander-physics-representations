"""Cross-config comparison: statistical comparison of agent variants.

Compares performance and behavior across agent configurations (labeled vs
blind vs history) using episode-level metrics from dedicated evaluation runs.
Produces comparison tables, grouped bar charts, and statistical test results.

Statistical approach:
  - Unit of analysis = per-seed means (not individual episodes).
    Episodes within a seed share trained weights and aren't independent.
  - Primary test: Mann-Whitney U (non-parametric, honest about small N).
  - Effect size: Cohen's d (mean difference / pooled std).
  - Multiple comparisons: Bonferroni correction reported but not enforced.

Design spec: analysis-tooling.md Section 6.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Metrics to compare
# ============================================================

# Primary performance metrics (computed from metrics.csv episode data).
# landed_pct etc. are derived from the 'outcome' column, not raw columns.
PERFORMANCE_METRICS = [
    "landed_pct",  # % episodes with outcome == "landed"
    "crashed_pct",  # % episodes with outcome == "crashed"
    "timeout_pct",  # % episodes with outcome == "timeout"
    "oob_pct",  # % episodes with outcome == "out_of_bounds"
    "mean_reward",  # mean total_reward across episodes
]

# Behavioral metrics (aggregated from per-episode columns in metrics.csv).
BEHAVIORAL_METRICS = [
    "fuel_efficiency",  # mean fuel_efficiency
    "thrust_duty_cycle",  # mean thrust_duty_cycle
    "mean_main_thrust",  # mean of mean_main_thrust
    "mean_abs_side_thrust",  # mean of mean_abs_side_thrust
    "std_main_thrust",  # variability in main thrust within episodes
    "std_side_thrust",  # variability in side thrust within episodes
    "main_thrust_frac_full",  # fraction of timesteps at full main thrust
    "main_thrust_frac_zero",  # fraction of timesteps with no main thrust
    "frac_descending",  # fraction of timesteps in descent phase
    "frac_hovering",  # fraction of timesteps hovering
    "frac_approaching",  # fraction of timesteps approaching pad
    "frac_correcting",  # fraction of timesteps correcting orientation
    "thrust_autocorr_lag1",  # temporal smoothness of main thrust
    "side_thrust_autocorr_lag1",  # temporal smoothness of side thrust
]

# Behavioral summary metrics (from behavioral_summary.json, per-seed).
BEHAVIORAL_SUMMARY_METRICS = [
    "adaptation_score",  # how much thrust pattern shifts across TWR quartiles
]


# ============================================================
# Data loading
# ============================================================


def load_comparison_metrics(
    comparison_data: dict,
) -> dict[str, list[pd.DataFrame]]:
    """Load per-seed metrics DataFrames for each variant in a comparison.

    Reads metrics.csv from each seed directory for each config (variant).
    Each DataFrame has one row per episode with all metric columns.

    Args:
        comparison_data: One comparison entry from the manifest, containing
            a 'configs' dict keyed by variant name. Each config has
            'seed_dirs' (list of resolved paths) and 'seeds' (list of ints).

    Returns:
        Dict mapping variant name -> list of per-seed DataFrames.
        Order matches the seed_dirs/seeds lists.

    Raises:
        FileNotFoundError: If any seed's metrics.csv doesn't exist.
    """
    result = {}

    for variant, config in comparison_data.get("configs", {}).items():
        seed_dfs = []
        # Support per-config trajectory_subdir override (default: "trajectories").
        # This allows corruption experiments to point at e.g. "trajectories-zero"
        # while uncorrupted configs use the default "trajectories".
        traj_subdir = config.get("trajectory_subdir", "trajectories")
        for seed_dir in config.get("seed_dirs", []):
            csv_path = Path(seed_dir) / traj_subdir / "metrics.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Missing metrics.csv for variant '{variant}': {csv_path}. "
                    f"Run the eval pipeline first."
                )
            df = pd.read_csv(csv_path)
            seed_dfs.append(df)
        result[variant] = seed_dfs

    return result


def load_behavioral_summaries(
    comparison_data: dict,
) -> dict[str, list[dict]]:
    """Load behavioral_summary.json from each seed for each variant.

    Args:
        comparison_data: Comparison entry from manifest.

    Returns:
        Dict mapping variant name -> list of behavioral summary dicts.
        Returns empty dict for seeds with missing summaries.
    """
    result = {}

    for variant, config in comparison_data.get("configs", {}).items():
        summaries = []
        traj_subdir = config.get("trajectory_subdir", "trajectories")
        for seed_dir in config.get("seed_dirs", []):
            json_path = (
                Path(seed_dir)
                / traj_subdir
                / "behavioral_analysis"
                / "behavioral_summary.json"
            )
            if json_path.exists():
                with open(json_path) as f:
                    summaries.append(json.load(f))
            else:
                summaries.append({})
        result[variant] = summaries

    return result


# ============================================================
# Per-seed aggregation
# ============================================================


def compute_variant_stats(seed_dfs: list[pd.DataFrame]) -> dict:
    """Compute per-seed aggregate metrics and cross-seed statistics.

    Each seed's episodes collapse to one number per metric.
    The per-seed means become the independent observations (N = len(seed_dfs)).

    This is the core aggregation step: 100 episodes/seed -> 1 value/seed,
    then N seed values -> mean +/- std. The per-seed values are the
    independent observations for statistical testing.

    Args:
        seed_dfs: List of per-seed DataFrames (one per seed).

    Returns:
        Dict mapping metric_name -> {mean, std, per_seed} where:
          - mean: mean of per-seed aggregates
          - std: std of per-seed aggregates (ddof=1 for unbiased estimate)
          - per_seed: list of per-seed values (the independent observations)
    """
    stats = {}

    for metrics in [PERFORMANCE_METRICS, BEHAVIORAL_METRICS]:
        for metric in metrics:
            per_seed_values = []
            for df in seed_dfs:
                # Performance metrics derived from outcome column.
                if metric == "landed_pct":
                    val = (df["outcome"] == "landed").mean() * 100
                elif metric == "crashed_pct":
                    val = (df["outcome"] == "crashed").mean() * 100
                elif metric == "timeout_pct":
                    val = (df["outcome"] == "timeout").mean() * 100
                elif metric == "oob_pct":
                    val = (df["outcome"] == "out_of_bounds").mean() * 100
                elif metric == "mean_reward":
                    val = df["total_reward"].mean()
                elif metric in df.columns:
                    # Behavioral metrics are direct column means.
                    val = df[metric].mean()
                else:
                    val = float("nan")
                per_seed_values.append(round(float(val), 4))

            valid = [v for v in per_seed_values if not np.isnan(v)]
            if valid:
                stats[metric] = {
                    "mean": round(float(np.mean(valid)), 4),
                    "std": (
                        round(float(np.std(valid, ddof=1)), 4)
                        if len(valid) > 1
                        else 0.0
                    ),
                    "per_seed": per_seed_values,
                }

    return stats


# ============================================================
# Statistical tests
# ============================================================


def run_statistical_tests(
    variant_stats: dict[str, dict],
    variant_names: list[str],
    metrics: list[str] | None = None,
) -> dict:
    """Run pairwise statistical tests between two variants.

    Uses Mann-Whitney U (non-parametric, appropriate for small N) and
    Cohen's d for effect size. With N=3 vs N=3, minimum possible
    p-value is 0.05 — this is honest about the data we have.

    Why Mann-Whitney U and not t-test:
      - With N=2-5 seeds per variant, we can't verify normality.
      - Mann-Whitney makes no distributional assumptions.
      - It's conservative: with N=3, min p=0.05 (can't over-claim).

    Why Cohen's d:
      - Reports practical significance alongside statistical significance.
      - |d| < 0.2 small, 0.2-0.8 medium, > 0.8 large.

    Args:
        variant_stats: Dict mapping variant_name -> output of compute_variant_stats().
        variant_names: Exactly two variant names to compare (e.g., ["labeled", "blind"]).
        metrics: Which metrics to test. Defaults to PERFORMANCE_METRICS + BEHAVIORAL_METRICS.

    Returns:
        Dict mapping metric_name -> {
            per_variant: {name: {mean, std, per_seed}},
            test: "mann_whitney_u",
            u_stat: float,
            p_value: float,
            effect_size_cohens_d: float,
            winner: str or "tie",
        }
    """
    from scipy.stats import mannwhitneyu

    if len(variant_names) != 2:
        raise ValueError(f"Expected exactly 2 variants, got {len(variant_names)}")

    if metrics is None:
        metrics = PERFORMANCE_METRICS + BEHAVIORAL_METRICS

    name_a, name_b = variant_names
    stats_a = variant_stats[name_a]
    stats_b = variant_stats[name_b]

    results = {}
    for metric in metrics:
        if metric not in stats_a or metric not in stats_b:
            continue

        values_a = stats_a[metric]["per_seed"]
        values_b = stats_b[metric]["per_seed"]

        # Filter NaN values.
        valid_a = [v for v in values_a if not np.isnan(v)]
        valid_b = [v for v in values_b if not np.isnan(v)]

        # Need at least 1 observation per group for any stats.
        if len(valid_a) < 1 or len(valid_b) < 1:
            continue

        # Mann-Whitney U test.
        # With very small N, mannwhitneyu may produce large p-values
        # but won't crash. This is the correct behavior — we're being
        # honest about insufficient statistical power.
        try:
            u_stat, p_value = mannwhitneyu(
                valid_a,
                valid_b,
                alternative="two-sided",
            )
        except ValueError:
            # Can happen with identical arrays or degenerate cases.
            u_stat, p_value = float("nan"), float("nan")

        # Cohen's d: (mean_a - mean_b) / pooled_std.
        # Absolute value since direction is captured in 'winner'.
        mean_a = np.mean(valid_a)
        mean_b = np.mean(valid_b)
        cohens_d = _cohens_d(valid_a, valid_b)

        # Determine winner based on metric direction.
        # For most metrics, higher is better.
        # For crash/timeout/OOB rates, lower is better.
        lower_is_better = metric in ("crashed_pct", "timeout_pct", "oob_pct")
        if mean_a == mean_b:
            winner = "tie"
        elif lower_is_better:
            winner = name_a if mean_a < mean_b else name_b
        else:
            winner = name_a if mean_a > mean_b else name_b

        results[metric] = {
            "per_variant": {
                name_a: {
                    "mean": round(float(mean_a), 4),
                    "std": stats_a[metric]["std"],
                    "per_seed": values_a,
                },
                name_b: {
                    "mean": round(float(mean_b), 4),
                    "std": stats_b[metric]["std"],
                    "per_seed": values_b,
                },
            },
            "test": "mann_whitney_u",
            "u_stat": (round(float(u_stat), 4) if not np.isnan(u_stat) else None),
            "p_value": (round(float(p_value), 4) if not np.isnan(p_value) else None),
            "effect_size_cohens_d": (
                round(float(cohens_d), 4) if not np.isnan(cohens_d) else None
            ),
            "winner": winner,
        }

    return results


def _cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation (assumes roughly equal variances).
    Returns NaN if either group has fewer than 2 observations
    (can't compute std with ddof=1).

    Formula: d = |mean_a - mean_b| / sqrt(pooled_variance)
    where pooled_var = ((n_a-1)*var_a + (n_b-1)*var_b) / (n_a+n_b-2)
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    # Pooled standard deviation.
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return float("nan")

    return abs(mean_a - mean_b) / pooled_std


# ============================================================
# Output writers
# ============================================================


def write_comparison_outputs(
    experiment_name: str,
    all_comparison_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Write all comparison output artifacts.

    Creates:
      - comparison_table.txt: human-readable results table (pasteable into notes)
      - comparison_table.csv: machine-readable results for further analysis
      - stat_tests.json: full statistical test results with per-seed values
      - {comparison_name}/metrics_by_variant.csv: per-episode data tagged by variant/seed

    Args:
        experiment_name: Human-readable experiment name (used in table header).
        all_comparison_results: Dict mapping comparison_name -> {
            variants, variant_stats, test_results, n_seeds, n_episodes,
            seed_dfs (optional, for per-comparison CSV),
            seeds (optional, seed IDs per variant)
        }.
        output_dir: Root directory for all outputs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_comparison_table_txt(experiment_name, all_comparison_results, out)
    _write_comparison_table_csv(all_comparison_results, out)
    _write_stat_tests_json(all_comparison_results, out)

    # Per-comparison outputs.
    for comp_name, comp_result in all_comparison_results.items():
        comp_dir = out / comp_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        # metrics_by_variant.csv: all episodes from all seeds, tagged.
        if "seed_dfs" in comp_result:
            _write_metrics_by_variant(comp_result, comp_dir)


def _write_comparison_table_txt(
    experiment_name: str,
    all_results: dict,
    out: Path,
) -> None:
    """Write human-readable comparison table.

    Format designed to be pasteable into research notes — fixed-width
    columns with aligned headers.
    """
    lines = [
        f"{experiment_name} -- Cross-Config Comparison",
        "=" * 90,
        (
            f"{'Comparison':<28s}  {'Variant':<10s}  {'N':>3s}  {'Eps':>5s}  "
            f"{'Landed%':>14s}  {'Reward':>14s}  {'p-value':>8s}"
        ),
        "-" * 90,
    ]

    for comp_name, result in sorted(all_results.items()):
        variants = result["variants"]
        test_results = result.get("test_results", {})
        variant_stats = result["variant_stats"]

        for i, variant in enumerate(variants):
            vs = variant_stats.get(variant, {})
            n_seeds = result["n_seeds"].get(variant, "?")
            n_eps = result["n_episodes"].get(variant, "?")

            landed = _fmt_mean_std(vs.get("landed_pct", {}))
            reward = _fmt_mean_std(vs.get("mean_reward", {}))

            # p-value only on the second variant (the comparison line).
            p_str = ""
            if i == 1 and "landed_pct" in test_results:
                p_val = test_results["landed_pct"].get("p_value")
                if p_val is not None:
                    sig = "*" if p_val < 0.05 else ""
                    p_str = f"{p_val:.3f}{sig}"

            comp_label = comp_name if i == 0 else ""
            lines.append(
                f"{comp_label:<28s}  {variant:<10s}  {n_seeds:>3}  {n_eps:>5}  "
                f"{landed:>14s}  {reward:>14s}  {p_str:>8s}"
            )

    (out / "comparison_table.txt").write_text("\n".join(lines) + "\n")


def _write_comparison_table_csv(all_results: dict, out: Path) -> None:
    """Write machine-readable comparison table as CSV."""
    fieldnames = [
        "comparison",
        "variant",
        "n_seeds",
        "n_episodes",
        "landed_pct_mean",
        "landed_pct_std",
        "mean_reward_mean",
        "mean_reward_std",
        "crashed_pct_mean",
        "timeout_pct_mean",
        "p_value_landed",
        "p_value_reward",
        "cohens_d_landed",
        "winner_landed",
    ]

    rows = []
    for comp_name, result in sorted(all_results.items()):
        test_results = result.get("test_results", {})

        for variant in result["variants"]:
            vs = result["variant_stats"].get(variant, {})
            rows.append(
                {
                    "comparison": comp_name,
                    "variant": variant,
                    "n_seeds": result["n_seeds"].get(variant, ""),
                    "n_episodes": result["n_episodes"].get(variant, ""),
                    "landed_pct_mean": _get_stat(vs, "landed_pct", "mean"),
                    "landed_pct_std": _get_stat(vs, "landed_pct", "std"),
                    "mean_reward_mean": _get_stat(vs, "mean_reward", "mean"),
                    "mean_reward_std": _get_stat(vs, "mean_reward", "std"),
                    "crashed_pct_mean": _get_stat(vs, "crashed_pct", "mean"),
                    "timeout_pct_mean": _get_stat(vs, "timeout_pct", "mean"),
                    "p_value_landed": test_results.get("landed_pct", {}).get(
                        "p_value", ""
                    ),
                    "p_value_reward": test_results.get("mean_reward", {}).get(
                        "p_value", ""
                    ),
                    "cohens_d_landed": test_results.get("landed_pct", {}).get(
                        "effect_size_cohens_d", ""
                    ),
                    "winner_landed": test_results.get("landed_pct", {}).get(
                        "winner", ""
                    ),
                }
            )

    with open(out / "comparison_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_stat_tests_json(all_results: dict, out: Path) -> None:
    """Write full statistical test results as JSON.

    Includes per-variant stats and per-seed values — the complete
    data for anyone wanting to verify or reanalyze the results.
    """
    stat_data = {}
    for comp_name, result in sorted(all_results.items()):
        stat_data[comp_name] = {
            "variants": result["variants"],
            **result.get("test_results", {}),
        }

    with open(out / "stat_tests.json", "w") as f:
        json.dump(stat_data, f, indent=2)


def _write_metrics_by_variant(comp_result: dict, comp_dir: Path) -> None:
    """Write per-comparison CSV with all episodes tagged by variant and seed.

    This is the raw data for custom analysis — every episode from every
    seed, with 'variant' and 'seed' columns added.
    """
    all_rows = []
    for variant, seed_dfs in comp_result["seed_dfs"].items():
        seeds = comp_result.get("seeds", {}).get(variant, range(len(seed_dfs)))
        for seed, df in zip(seeds, seed_dfs):
            df_copy = df.copy()
            df_copy["variant"] = variant
            df_copy["seed"] = seed
            all_rows.append(df_copy)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(comp_dir / "metrics_by_variant.csv", index=False)


def _fmt_mean_std(metric: dict) -> str:
    """Format {mean, std} dict as 'mean +/- std' string."""
    mean = metric.get("mean")
    std = metric.get("std")
    if mean is None:
        return "---"
    if std is not None and std > 0:
        return f"{mean:.1f} +/- {std:.1f}"
    return f"{mean:.1f}"


def _get_stat(stats: dict, metric_name: str, stat_name: str):
    """Safely extract a stat value from nested stats dict."""
    return stats.get(metric_name, {}).get(stat_name, "")


# ============================================================
# Plot generation
# ============================================================


def plot_performance_bars(
    all_comparison_results: dict[str, dict],
    output_path: str,
) -> None:
    """Grouped bar chart of performance metrics across comparisons.

    X-axis = comparisons (one group per condition×difficulty).
    Bars = variants (labeled, blind, etc.).
    Two panels: landed% (0-100 scale) and mean reward.
    Error bars from cross-seed std. Individual seed dots overlaid.

    Args:
        all_comparison_results: Dict mapping comparison_name -> result dict.
        output_path: Where to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp_names = sorted(all_comparison_results.keys())
    if not comp_names:
        return

    # Collect all variant names across comparisons.
    all_variants = []
    for r in all_comparison_results.values():
        for v in r["variants"]:
            if v not in all_variants:
                all_variants.append(v)

    # Consistent colors for known variants.
    variant_colors = {
        "labeled": "#2196F3",
        "blind": "#4CAF50",
        "history": "#FF9800",
    }
    default_colors = ["#9C27B0", "#F44336", "#00BCD4", "#795548"]

    fig, axes = plt.subplots(1, 2, figsize=(max(6, 3 * len(comp_names)), 6))

    metrics_to_plot = [
        ("landed_pct", "Landed %", (0, 100)),
        ("mean_reward", "Mean Reward", None),
    ]

    for ax, (metric, ylabel, ylim) in zip(axes, metrics_to_plot):
        _draw_grouped_bars(
            ax,
            comp_names,
            all_variants,
            all_comparison_results,
            metric,
            ylabel,
            variant_colors,
            default_colors,
            ylim=ylim,
        )

    fig.suptitle("Performance Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_behavioral_comparison(
    all_comparison_results: dict[str, dict],
    output_path: str,
) -> None:
    """Grouped bar chart of behavioral metrics across comparisons.

    Same layout as performance_bars but for behavioral metrics:
    fuel efficiency, thrust duty cycle, mean main thrust.

    Args:
        all_comparison_results: Dict mapping comparison_name -> result dict.
        output_path: Where to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp_names = sorted(all_comparison_results.keys())
    if not comp_names:
        return

    all_variants = []
    for r in all_comparison_results.values():
        for v in r["variants"]:
            if v not in all_variants:
                all_variants.append(v)

    variant_colors = {
        "labeled": "#2196F3",
        "blind": "#4CAF50",
        "history": "#FF9800",
    }
    default_colors = ["#9C27B0", "#F44336", "#00BCD4", "#795548"]

    metrics_to_plot = [
        ("fuel_efficiency", "Fuel Efficiency", None),
        ("thrust_duty_cycle", "Thrust Duty Cycle", None),
        ("mean_main_thrust", "Mean Main Thrust", None),
        ("std_main_thrust", "Std Main Thrust", None),
        ("main_thrust_frac_zero", "Frac Zero Thrust", None),
        ("main_thrust_frac_full", "Frac Full Thrust", None),
        ("frac_descending", "Frac Descending", None),
        ("frac_correcting", "Frac Correcting", None),
        ("thrust_autocorr_lag1", "Thrust Autocorr", None),
        ("side_thrust_autocorr_lag1", "Side Autocorr", None),
    ]

    n_metrics = len(metrics_to_plot)
    n_cols = min(5, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(6, 3 * len(comp_names)) * n_cols / 3, 6 * n_rows),
    )
    axes = np.array(axes).flatten()

    for ax, (metric, ylabel, ylim) in zip(axes, metrics_to_plot):
        _draw_grouped_bars(
            ax,
            comp_names,
            all_variants,
            all_comparison_results,
            metric,
            ylabel,
            variant_colors,
            default_colors,
            ylim=ylim,
        )

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Behavioral Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_breakdown(
    variant_stats: dict[str, dict],
    variant_names: list[str],
    comparison_name: str,
    output_path: str,
) -> None:
    """Stacked bar chart of outcome breakdown per variant.

    Shows landed/crashed/timeout/OOB proportions for each variant.
    Useful for understanding failure modes — does one variant crash
    more while another times out?

    Args:
        variant_stats: Dict mapping variant_name -> stats.
        variant_names: Variant names for ordering.
        comparison_name: Used in plot title.
        output_path: Where to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcome_metrics = [
        ("landed_pct", "Landed", "#4CAF50"),
        ("crashed_pct", "Crashed", "#F44336"),
        ("timeout_pct", "Timeout", "#FF9800"),
        ("oob_pct", "OOB", "#9C27B0"),
    ]

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(variant_names)), 5))

    x = np.arange(len(variant_names))
    bottoms = np.zeros(len(variant_names))

    for metric, label, color in outcome_metrics:
        values = []
        for variant in variant_names:
            vs = variant_stats.get(variant, {})
            values.append(vs.get(metric, {}).get("mean", 0))

        ax.bar(x, values, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, fontsize=11)
    ax.set_ylabel("Percentage", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.set_title(f"Outcome Breakdown — {comparison_name}", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_grouped_bars(
    ax,
    comp_names,
    all_variants,
    all_results,
    metric,
    ylabel,
    variant_colors,
    default_colors,
    ylim=None,
):
    """Draw grouped bars for one metric across comparisons.

    Each comparison gets a group of bars (one per variant).
    Error bars from cross-seed std. Individual seed values overlaid
    as scatter dots for transparency about N.
    """
    n_comps = len(comp_names)
    n_variants = len(all_variants)
    bar_width = 0.8 / n_variants
    x = np.arange(n_comps)

    for j, variant in enumerate(all_variants):
        color = variant_colors.get(variant, default_colors[j % len(default_colors)])
        means = []
        stds = []
        seed_points = []

        for comp_name in comp_names:
            result = all_results[comp_name]
            vs = result["variant_stats"].get(variant, {})
            m = vs.get(metric, {})
            means.append(m.get("mean", 0))
            stds.append(m.get("std", 0))
            seed_points.append(m.get("per_seed", []))

        offset = (j - n_variants / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width * 0.9,
            yerr=stds,
            capsize=3,
            label=variant,
            color=color,
            alpha=0.8,
            edgecolor="white",
        )

        # Overlay individual seed dots for transparency.
        for i, seeds in enumerate(seed_points):
            if seeds:
                # Small jitter to avoid overlap.
                rng = np.random.RandomState(42 + j)
                jitter = rng.uniform(-bar_width * 0.2, bar_width * 0.2, len(seeds))
                ax.scatter(
                    [x[i] + offset + jx for jx in jitter],
                    seeds,
                    s=25,
                    color="black",
                    alpha=0.5,
                    zorder=5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
