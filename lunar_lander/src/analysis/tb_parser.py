"""TensorBoard event file parser for seed aggregation.

Wraps TensorBoard's EventAccumulator into a clean dict-based API.
Two main functions:

  - parse_tb_events(log_dir) -> dict of tag -> [(step, value), ...]
  - extract_last_k_metrics(scalars, tags, last_k) -> dict of tag -> float

The parser reads ALL scalar events from a training run directory.
extract_last_k_metrics() then computes end-of-training values as the mean
of the last K eval checkpoints — this smooths single-evaluation noise and
matches the "mean last 5" convention used in training-runs.md.

NOTE: These are training-time metrics (from TB eval callbacks), NOT proper
post-training evaluation. For definitive performance numbers, run dedicated
eval episodes with eval_agent.py.

Used by aggregate_seeds.py to read training metrics from multiple seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_tb_events(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Parse all scalar events from a TensorBoard log directory.

    Reads the events.out.tfevents.* files in log_dir using TensorBoard's
    EventAccumulator. Returns a dict mapping each scalar tag (e.g.,
    "eval/landed_pct") to a sorted list of (step, value) tuples.

    Args:
        log_dir: Path to directory containing TensorBoard event files.
            This is typically the training run directory (same level
            as model.zip and config.json).

    Returns:
        Dict mapping tag names to lists of (global_step, scalar_value)
        tuples, sorted by step ascending. Empty dict if no events found.

    Raises:
        FileNotFoundError: If log_dir does not exist.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"TB log directory not found: {log_dir}")

    # Lazy import — tensorboard is heavy and only needed here.
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    # size_guidance=0 means "load all events" (default caps at 10K).
    # We need full history for learning curve plots.
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()

    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        # Sort by step — events are usually ordered, but be safe.
        pairs = sorted([(e.step, e.value) for e in events], key=lambda x: x[0])
        scalars[tag] = pairs

    return scalars


def extract_last_k_metrics(
    scalars: dict[str, list[tuple[int, float]]],
    tags: list[str],
    last_k: int = 5,
) -> dict[str, float]:
    """Extract end-of-training metric values by averaging the last K checkpoints.

    For each requested tag, takes the mean of the last K values in the
    scalar time series. This smooths noise from single-evaluation variance —
    an agent at 88% landed might score 82% or 94% on any single eval.

    These are training-time estimates only. For definitive evaluation, use
    dedicated episode collection (eval_agent.py).

    If fewer than K checkpoints exist, all values are used.
    If the tag is missing entirely, returns NaN.

    Args:
        scalars: Output from parse_tb_events().
        tags: List of tag names to extract (e.g., ["eval/landed_pct"]).
        last_k: Number of final checkpoints to average. Default 5.

    Returns:
        Dict mapping each tag to its averaged value (or NaN).
    """
    result: dict[str, float] = {}
    for tag in tags:
        if tag not in scalars or len(scalars[tag]) == 0:
            result[tag] = float("nan")
            continue
        values = [v for _, v in scalars[tag]]
        # Take last K values (or all if fewer than K).
        tail = values[-last_k:]
        result[tag] = float(np.mean(tail))
    return result
