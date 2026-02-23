"""Seed aggregation: cross-seed statistics from TensorBoard training metrics.

Core aggregation logic used by aggregate_seeds.py. Given a list of seed
directories for one config, reads TB events from each, computes aligned
learning curves and training metric statistics, and checks seed consistency.

NOTE: All metrics here come from TensorBoard training logs (eval callbacks),
NOT from dedicated post-training evaluation runs. For definitive performance
numbers, use eval_agent.py with proper episode collection.

Three output dicts per config:
  - training_metrics: mean/std/median/per_seed for key eval metrics
  - learning_curves: per-step means and per-seed traces
  - seed_consistency: CV and outlier flags

Design spec: analysis-tooling.md Section 5.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lunar_lander.src.analysis.tb_parser import (
    extract_last_k_metrics,
    parse_tb_events,
)


# --- Metric tag lists ---
# These define which TensorBoard tags we aggregate and how.

# Eval metrics to aggregate — these are the ones that matter for
# comparing agent performance across seeds.
EVAL_METRIC_TAGS = [
    "eval/landed_pct",
    "eval/crashed_pct",
    "eval/timeout_pct",
    "eval/out_of_bounds_pct",
    "eval/mean_reward",
]

# Training dynamics metrics — useful for diagnosing why seeds differ.
TRAIN_METRIC_TAGS = [
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/approx_kl",
]

# Learning curve metrics — the subset we track over training steps.
# Eval metrics (logged every eval_freq steps).
CURVE_EVAL_TAGS = [
    "eval/landed_pct",
    "eval/mean_reward",
]
# Training dynamics (logged every n_steps rollout).
CURVE_TRAIN_TAGS = [
    "train/entropy_loss",
    "train/value_loss",
    "train/approx_kl",
]
# Combined for the aggregation computation.
CURVE_METRIC_TAGS = CURVE_EVAL_TAGS + CURVE_TRAIN_TAGS

# --- Consistency thresholds ---
# CV above this flags the config as having high cross-seed variance.
CV_THRESHOLD = 0.15
# Outlier threshold — seed more than this many std from mean is flagged.
OUTLIER_STD_THRESHOLD = 2.0


def _short_name(tag: str) -> str:
    """Strip prefix from TB tag: 'eval/landed_pct' -> 'landed_pct'."""
    return tag.split("/", 1)[-1]


# ============================================================
# Core aggregation
# ============================================================


def aggregate_seed_metrics(
    seed_dirs: list[str],
    seeds: list[int],
    last_k: int = 5,
) -> dict:
    """Aggregate training metrics across multiple seeds for one config.

    Reads TensorBoard events from each seed directory, computes final
    metric statistics (mean/std/median across seeds), aligned learning
    curves, and seed consistency checks.

    Args:
        seed_dirs: List of paths to seed run directories (each containing
            TensorBoard event files). Order must match seeds list.
        seeds: List of seed numbers corresponding to seed_dirs.
        last_k: Number of final eval checkpoints to average for
            "final" metrics. Default 5.

    Returns:
        Dict with four keys:
          - "training_metrics": {metric_name: {mean, std, median, per_seed}}
          - "training_dynamics": {metric_name: {mean, std, per_seed}}
          - "learning_curves": {steps, metric_name: {mean, std, per_seed}}
          - "seed_consistency": {metric_cv, consistent, flags}
    """
    assert len(seed_dirs) == len(seeds), "seed_dirs and seeds must match"

    # --- Phase 1: Read TB events from all seeds ---
    # Each seed produces a dict of tag -> [(step, value), ...].
    all_scalars = {}
    for seed, seed_dir in zip(seeds, seed_dirs):
        all_scalars[seed] = parse_tb_events(seed_dir)

    # --- Phase 2: Final metrics (mean of last K checkpoints per seed) ---
    # We extract all eval + train tags, then separate for output.
    all_tags = EVAL_METRIC_TAGS + TRAIN_METRIC_TAGS
    per_seed_finals = {}
    for seed in seeds:
        per_seed_finals[seed] = extract_last_k_metrics(
            all_scalars[seed],
            tags=all_tags,
            last_k=last_k,
        )

    # Build cross-seed statistics for eval metrics.
    training_metrics = {}
    for tag in EVAL_METRIC_TAGS:
        name = _short_name(tag)
        values = [per_seed_finals[s][tag] for s in seeds]
        # Filter out NaN (missing seeds) for stats.
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            training_metrics[name] = {
                "mean": round(float(np.mean(valid)), 2),
                "std": round(
                    float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0, 2
                ),
                "median": round(float(np.median(valid)), 2),
                "per_seed": [round(v, 2) if not np.isnan(v) else None for v in values],
            }
        else:
            training_metrics[name] = {
                "mean": None,
                "std": None,
                "median": None,
                "per_seed": [None] * len(seeds),
            }

    # Training dynamics (final values only, simpler structure).
    training_dynamics = {}
    for tag in TRAIN_METRIC_TAGS:
        name = "final_" + _short_name(tag)
        values = [per_seed_finals[s].get(tag, float("nan")) for s in seeds]
        valid = [v for v in values if not np.isnan(v)]
        if valid:
            training_dynamics[name] = {
                "mean": round(float(np.mean(valid)), 4),
                "std": round(
                    float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0, 4
                ),
                "per_seed": [round(v, 4) if not np.isnan(v) else None for v in values],
            }

    # --- Phase 3: Learning curves (aligned across seeds) ---
    learning_curves = _compute_learning_curves(all_scalars, seeds)

    # --- Phase 4: Seed consistency ---
    seed_consistency = _compute_seed_consistency(training_metrics)

    return {
        "training_metrics": training_metrics,
        "training_dynamics": training_dynamics,
        "learning_curves": learning_curves,
        "seed_consistency": seed_consistency,
    }


def _compute_learning_curves(
    all_scalars: dict[int, dict],
    seeds: list[int],
) -> dict:
    """Align learning curves across seeds and compute per-step statistics.

    Seeds may have slightly different step counts. We use the intersection
    of steps across all seeds for alignment. Eval and train tags may log
    at different frequencies, so each metric gets its own "steps" array.

    Returns dict with per-metric {steps, mean, std, per_seed} sub-dicts.
    Also includes a top-level "steps" key with eval steps for backwards
    compatibility.
    """
    curves: dict = {"steps": []}

    for tag in CURVE_METRIC_TAGS:
        name = _short_name(tag)

        # Collect step->value for each seed.
        seed_data = {}
        for seed in seeds:
            if tag in all_scalars[seed]:
                seed_data[seed] = dict(all_scalars[seed][tag])
            else:
                seed_data[seed] = {}

        # Find common steps (intersection of all seeds' checkpoints).
        if seed_data:
            step_sets = [set(d.keys()) for d in seed_data.values() if d]
            if step_sets:
                common_steps = sorted(set.intersection(*step_sets))
            else:
                common_steps = []
        else:
            common_steps = []

        # Top-level "steps" uses eval steps (first eval tag encountered).
        if not curves["steps"] and common_steps and tag in CURVE_EVAL_TAGS:
            curves["steps"] = common_steps

        # Build aligned arrays.
        if common_steps:
            per_seed = {}
            for seed in seeds:
                per_seed[str(seed)] = [
                    round(seed_data[seed].get(step, float("nan")), 4)
                    for step in common_steps
                ]
            values_matrix = np.array([per_seed[str(s)] for s in seeds])
            curves[name] = {
                "steps": common_steps,
                "mean": [round(float(v), 4) for v in np.nanmean(values_matrix, axis=0)],
                "std": (
                    [
                        round(float(v), 4)
                        for v in np.nanstd(values_matrix, axis=0, ddof=1)
                    ]
                    if len(seeds) > 1
                    else [0.0] * len(common_steps)
                ),
                "per_seed": per_seed,
            }
        else:
            curves[name] = {"steps": [], "mean": [], "std": [], "per_seed": {}}

    return curves


def _compute_seed_consistency(training_metrics: dict) -> dict:
    """Check whether seeds tell a consistent story.

    Flags inconsistency if:
      - Coefficient of variation (CV = std/mean) exceeds threshold for
        any key metric (landed_pct, mean_reward).
      - Any seed is an outlier (>2 std from mean) on key metrics.
    """
    key_metrics = ["landed_pct", "mean_reward"]
    consistency: dict = {"consistent": True, "flags": []}

    for name in key_metrics:
        if name not in training_metrics or training_metrics[name]["mean"] is None:
            continue

        mean = training_metrics[name]["mean"]
        std = training_metrics[name]["std"]
        per_seed = training_metrics[name]["per_seed"]

        # CV check (only meaningful if mean != 0).
        if mean != 0 and std is not None:
            cv = abs(std / mean)
            consistency[f"{name}_cv"] = round(cv, 4)

            if cv > CV_THRESHOLD:
                consistency["consistent"] = False
                consistency["flags"].append(
                    f"{name}: CV={cv:.3f} exceeds threshold {CV_THRESHOLD}"
                )

        # Outlier check.
        if std is not None and std > 0:
            for i, val in enumerate(per_seed):
                if val is not None and abs(val - mean) > OUTLIER_STD_THRESHOLD * std:
                    consistency["consistent"] = False
                    consistency["flags"].append(
                        f"{name}: seed index {i} value {val} is "
                        f">{OUTLIER_STD_THRESHOLD} std from mean {mean}"
                    )

    return consistency


# ============================================================
# Output writers
# ============================================================


def write_config_outputs(
    config_name: str,
    config_data: dict,
    agg_result: dict,
    output_dir: str,
) -> None:
    """Write per-config aggregation artifacts to disk.

    Creates three JSON files in output_dir:
      - metrics.json: final metrics + metadata (the representative result)
      - learning_curves.json: step-aligned curves for plotting
      - seed_consistency.json: CV and outlier flags

    Args:
        config_name: Config identifier (e.g., "blind-ppo-easy-128-lowent").
        config_data: Config metadata from manifest (variant, condition, etc.).
        agg_result: Output from aggregate_seed_metrics().
        output_dir: Directory to write artifacts into (created if needed).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # metrics.json — the representative result for this config.
    metrics = {
        "config_name": config_name,
        "variant": config_data.get("variant"),
        "condition": config_data.get("condition"),
        "profile": config_data.get("profile"),
        "net_arch": config_data.get("net_arch"),
        "ent_coef": config_data.get("ent_coef"),
        "n_seeds": len(config_data.get("seeds", [])),
        "seeds": config_data.get("seeds", []),
        "training_metrics": agg_result["training_metrics"],
        "training_dynamics": agg_result.get("training_dynamics", {}),
    }
    _write_json(out / "metrics.json", metrics)

    # learning_curves.json — for plotting.
    _write_json(out / "learning_curves.json", agg_result["learning_curves"])

    # seed_consistency.json — are seeds telling the same story?
    _write_json(out / "seed_consistency.json", agg_result["seed_consistency"])


def write_experiment_summary(
    experiment_name: str,
    configs_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Write experiment-level summary table (txt + csv).

    Args:
        experiment_name: Human-readable experiment name.
        configs_results: Dict mapping config_name -> {n_seeds, training_metrics}.
        output_dir: Directory to write summary files into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- summary_table.txt ---
    lines = [
        f"{experiment_name} -- Seed Aggregation",
        "=" * 72,
        f"{'Config':<50s}  {'N':>3s}  {'Landed%':>12s}  {'Reward':>12s}",
        "-" * 72,
    ]
    for config_name, result in sorted(configs_results.items()):
        n = result.get("n_seeds", "?")
        fm = result.get("training_metrics", {})
        landed = _fmt_mean_std(fm.get("landed_pct", {}))
        reward = _fmt_mean_std(fm.get("mean_reward", {}))
        lines.append(f"{config_name:<50s}  {n:>3}  {landed:>12s}  {reward:>12s}")

    (out / "summary_table.txt").write_text("\n".join(lines) + "\n")

    # --- summary_table.csv ---
    fieldnames = [
        "config",
        "n_seeds",
        "landed_pct_mean",
        "landed_pct_std",
        "mean_reward_mean",
        "mean_reward_std",
        "crashed_pct_mean",
        "timeout_pct_mean",
    ]
    rows = []
    for config_name, result in sorted(configs_results.items()):
        fm = result.get("training_metrics", {})
        rows.append(
            {
                "config": config_name,
                "n_seeds": result.get("n_seeds", ""),
                "landed_pct_mean": _get_val(fm, "landed_pct", "mean"),
                "landed_pct_std": _get_val(fm, "landed_pct", "std"),
                "mean_reward_mean": _get_val(fm, "mean_reward", "mean"),
                "mean_reward_std": _get_val(fm, "mean_reward", "std"),
                "crashed_pct_mean": _get_val(fm, "crashed_pct", "mean"),
                "timeout_pct_mean": _get_val(fm, "timeout_pct", "mean"),
            }
        )

    with open(out / "summary_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Plot generation
# ============================================================


def plot_learning_curves(
    learning_curves: dict,
    seeds: list[int],
    config_name: str,
    output_path: str,
) -> None:
    """Plot learning curves: individual seed traces + aggregated mean.

    Two rows:
      - Top row: eval metrics (reward, landed%) — share x-axis
      - Bottom row: training dynamics (entropy, value loss, approx KL) — share x-axis

    Eval and train metrics may have different step frequencies, so each
    row uses its own x-axis from the metric's own "steps" array.

    Individual seeds as thin lines, mean as thicker line.

    Args:
        learning_curves: Output from aggregate_seed_metrics()["learning_curves"].
        seeds: List of seed numbers.
        config_name: Used in plot title.
        output_path: Where to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # Define what to plot in each row.
    eval_metrics = [
        ("mean_reward", "Reward", "#2196F3"),
        ("landed_pct", "Landed %", "#4CAF50"),
    ]
    train_metrics = [
        ("entropy_loss", "Entropy Loss", "#795548"),
        ("value_loss", "Value Loss", "#E91E63"),
        ("approx_kl", "Approx KL", "#00BCD4"),
    ]

    # Filter to metrics that exist in the data.
    eval_metrics = [(k, l, c) for k, l, c in eval_metrics if k in learning_curves]
    train_metrics = [(k, l, c) for k, l, c in train_metrics if k in learning_curves]

    n_eval = len(eval_metrics)
    n_train = len(train_metrics)

    if n_eval == 0 and n_train == 0:
        return  # Nothing to plot.

    n_rows = n_eval + n_train
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), squeeze=False)
    axes = axes[:, 0]  # Flatten from (n, 1) to (n,).

    def _step_formatter(x, pos):
        if x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x / 1_000:.0f}K"
        return str(int(x))

    row = 0

    # --- Eval metrics ---
    for metric_name, ylabel, base_color in eval_metrics:
        ax = axes[row]
        mc = learning_curves[metric_name]
        # Each metric has its own steps array.
        metric_steps = mc.get("steps", learning_curves.get("steps", []))

        if not metric_steps:
            row += 1
            continue

        # Individual seed traces — thin, semi-transparent.
        per_seed = mc.get("per_seed", {})
        for seed in seeds:
            seed_key = str(seed)
            if seed_key in per_seed:
                ax.plot(
                    metric_steps,
                    per_seed[seed_key],
                    linewidth=0.8,
                    alpha=0.4,
                    color=base_color,
                )

        # Aggregated mean.
        if mc.get("mean"):
            ax.plot(
                metric_steps, mc["mean"], linewidth=2.0, color=base_color, label="mean"
            )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
        row += 1

    # --- Training dynamics ---
    for metric_name, ylabel, base_color in train_metrics:
        ax = axes[row]
        mc = learning_curves[metric_name]
        metric_steps = mc.get("steps", [])

        if not metric_steps:
            row += 1
            continue

        per_seed = mc.get("per_seed", {})
        for seed in seeds:
            seed_key = str(seed)
            if seed_key in per_seed:
                ax.plot(
                    metric_steps,
                    per_seed[seed_key],
                    linewidth=0.8,
                    alpha=0.4,
                    color=base_color,
                )

        if mc.get("mean"):
            ax.plot(
                metric_steps, mc["mean"], linewidth=2.0, color=base_color, label="mean"
            )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(_step_formatter))
        row += 1

    axes[-1].set_xlabel("Training Steps", fontsize=11)
    fig.suptitle(f"Learning Curves -- {config_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    training_metrics: dict,
    seeds: list[int],
    config_name: str,
    output_path: str,
    training_dynamics: dict | None = None,
) -> None:
    """Plot final metrics overview: strip/swarm of per-seed values.

    Two rows:
      - Top row: all eval metrics (landed%, crashed%, timeout%, OOB%, reward)
      - Bottom row: training dynamics (entropy, policy grad, value loss, approx KL)

    At small N, show all data points — raw points are more informative
    than summary statistics.

    Args:
        training_metrics: Output from aggregate_seed_metrics()["training_metrics"].
        seeds: List of seed numbers.
        config_name: Used in plot title.
        output_path: Where to save the PNG.
        training_dynamics: Output from aggregate_seed_metrics()["training_dynamics"].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Top row: eval metrics. Bottom row: training dynamics.
    # Percentage metrics get fixed 0-100 y-axis so relative magnitudes
    # are visually comparable across configs. Reward and dynamics auto-scale.
    _PCT_YLIM = (0, 100)

    eval_panels = [
        ("landed_pct", "Landed %", "#4CAF50", _PCT_YLIM),
        ("crashed_pct", "Crashed %", "#F44336", _PCT_YLIM),
        ("timeout_pct", "Timeout %", "#FF9800", _PCT_YLIM),
        ("out_of_bounds_pct", "OOB %", "#9C27B0", _PCT_YLIM),
        ("mean_reward", "Mean Reward", "#2196F3", None),
    ]

    dynamics_panels = [
        ("final_entropy_loss", "Entropy Loss", "#795548", None),
        ("final_policy_gradient_loss", "Policy Grad Loss", "#607D8B", None),
        ("final_value_loss", "Value Loss", "#E91E63", None),
        ("final_approx_kl", "Approx KL", "#00BCD4", None),
    ]

    # Filter to panels that have data.
    eval_panels = [
        (k, l, c, yl) for k, l, c, yl in eval_panels if k in training_metrics
    ]
    if training_dynamics:
        dynamics_panels = [
            (k, l, c, yl) for k, l, c, yl in dynamics_panels if k in training_dynamics
        ]
    else:
        dynamics_panels = []

    n_eval = len(eval_panels)
    n_dyn = len(dynamics_panels)
    has_dynamics = n_dyn > 0

    if n_eval == 0:
        return  # Nothing to plot.

    # Layout: 1 or 2 rows.
    if has_dynamics:
        n_cols = max(n_eval, n_dyn)
        fig, all_axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 8))
        eval_axes = all_axes[0]
        dyn_axes = all_axes[1]
        # Hide unused columns in each row.
        for i in range(n_eval, n_cols):
            eval_axes[i].set_visible(False)
        for i in range(n_dyn, n_cols):
            dyn_axes[i].set_visible(False)
    else:
        fig, eval_axes = plt.subplots(1, n_eval, figsize=(3.2 * n_eval, 5))
        if n_eval == 1:
            eval_axes = [eval_axes]

    # --- Draw eval panels ---
    for ax, (metric_name, label, color, ylim) in zip(eval_axes, eval_panels):
        _draw_dot_panel(
            ax, training_metrics[metric_name], seeds, label, color, ylim=ylim
        )

    # --- Draw dynamics panels ---
    if has_dynamics:
        for ax, (metric_name, label, color, ylim) in zip(dyn_axes, dynamics_panels):
            _draw_dot_panel(
                ax, training_dynamics[metric_name], seeds, label, color, ylim=ylim
            )

    fig.suptitle(f"Training Metrics -- {config_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_dot_panel(
    ax,
    metric_data: dict,
    seeds: list[int],
    label: str,
    color: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw a single dot-strip panel for one metric.

    Shows per-seed values as dots with a dashed mean line.
    Used by plot_summary for both eval metrics and training dynamics.

    Args:
        ylim: Fixed y-axis range, e.g. (0, 100) for percentages.
              None means auto-scale to data.
    """
    per_seed = metric_data.get("per_seed", [])
    mean_val = metric_data.get("mean")

    # Scatter plot of per-seed values.
    valid_points = [(i, v) for i, v in enumerate(per_seed) if v is not None]
    if valid_points:
        xs, ys = zip(*valid_points)
        # Jitter x slightly for visibility at small N.
        jittered_x = [1.0 + np.random.uniform(-0.1, 0.1) for _ in xs]
        ax.scatter(
            jittered_x,
            ys,
            s=80,
            c=color,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            zorder=3,
        )

        # Label each point with seed number.
        for jx, y, idx in zip(jittered_x, ys, xs):
            ax.annotate(
                f"s{seeds[idx]}",
                (jx, y),
                textcoords="offset points",
                xytext=(8, 0),
                fontsize=7,
                color="gray",
            )

    # Mean as horizontal line.
    if mean_val is not None:
        ax.axhline(
            mean_val,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=f"mean={mean_val:.4g}",
        )
        ax.legend(fontsize=8)

    ax.set_ylabel(label, fontsize=10)
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, axis="y", alpha=0.3)


# ============================================================
# Internal helpers
# ============================================================


def _write_json(path: Path, data: dict) -> None:
    """Write dict to JSON file with readable formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _fmt_mean_std(metric: dict) -> str:
    """Format a metric dict as 'mean +/- std' string."""
    mean = metric.get("mean")
    std = metric.get("std")
    if mean is None:
        return "---"
    if std is not None and std > 0:
        return f"{mean:.1f} +/- {std:.1f}"
    return f"{mean:.1f}"


def _get_val(training_metrics: dict, metric_name: str, stat: str):
    """Safely get a stat from nested training_metrics."""
    m = training_metrics.get(metric_name, {})
    return m.get(stat, "")
