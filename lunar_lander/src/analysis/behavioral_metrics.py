"""Per-episode action distribution histograms for Lunar Lander.

Computes action histograms from .npz trajectory files. These are
the Level 1 (per-step) data foundation for all distributional
analysis — adaptation scoring, behavioral fingerprints, and
cross-model comparisons all start here.

Main entry points:

    compute_action_histograms(npz_path, n_bins=50) -> dict
        Single episode — returns histogram counts and bin edges
        for main thrust and side thrust.

    compute_collection_histograms(directory, workers=8) -> dict
        All .npz files in a directory — returns dict mapping
        npz_path -> histogram dict. Supports parallel computation.
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np


# Default number of histogram bins. 50 gives ~0.02 resolution on
# main thrust [0,1] and ~0.04 on side thrust [-1,1]. Fine enough
# to distinguish bang-bang from proportional control.
DEFAULT_N_BINS = 50


def compute_action_histograms(
    npz_path: str | Path,
    n_bins: int = DEFAULT_N_BINS,
) -> dict:
    """Compute action distribution histograms from a single episode.

    Loads the .npz file, extracts the actions array, and computes
    histograms over fixed domains:
        main_thrust: [0, 1]  (continuous throttle)
        side_thrust: [-1, 1] (continuous lateral control)

    Using fixed domains (not data-adaptive edges) ensures histograms
    from different episodes are directly comparable — same bin edges,
    same meaning per bin.

    Args:
        npz_path: Path to the .npz trajectory file.
        n_bins: Number of histogram bins (default 50).

    Returns:
        Dict with keys:
            main_thrust_counts: (n_bins,) int array — bin counts
            main_thrust_edges:  (n_bins+1,) float array — bin edges [0, 1]
            side_thrust_counts: (n_bins,) int array — bin counts
            side_thrust_edges:  (n_bins+1,) float array — bin edges [-1, 1]
    """
    data = np.load(str(npz_path), allow_pickle=False)
    actions = data["actions"]  # (T, 2)

    # Main thrust histogram over [0, 1].
    main_counts, main_edges = np.histogram(actions[:, 0], bins=n_bins, range=(0.0, 1.0))

    # Side thrust histogram over [-1, 1].
    side_counts, side_edges = np.histogram(
        actions[:, 1], bins=n_bins, range=(-1.0, 1.0)
    )

    return {
        "main_thrust_counts": main_counts,
        "main_thrust_edges": main_edges,
        "side_thrust_counts": side_counts,
        "side_thrust_edges": side_edges,
    }


def _compute_histograms_worker(args: tuple) -> tuple[str, dict]:
    """Worker function for parallel histogram computation.

    Wraps compute_action_histograms to accept (npz_path, n_bins)
    tuple and return (npz_path, result) tuple — required by Pool.map.
    """
    npz_path, n_bins = args
    return npz_path, compute_action_histograms(npz_path, n_bins=n_bins)


def compute_collection_histograms(
    directory: str | Path,
    workers: int = 8,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, dict]:
    """Compute histograms for all .npz files in a directory.

    Finds all .npz files, computes per-episode histograms (optionally
    in parallel), and returns a dict mapping npz_path to histogram dict.

    Args:
        directory: Path to directory containing .npz files.
        workers: Number of parallel workers. Use 1 for sequential.
        n_bins: Number of histogram bins.

    Returns:
        Dict mapping npz_path (str) -> histogram dict (from
        compute_action_histograms).
    """
    npz_paths = sorted(Path(directory).glob("*.npz"))
    if not npz_paths:
        return {}

    npz_strs = [str(p) for p in npz_paths]
    args = [(p, n_bins) for p in npz_strs]

    if workers <= 1:
        # Sequential — simpler for debugging.
        results = [_compute_histograms_worker(a) for a in args]
    else:
        # Parallel — each worker loads one .npz and computes histograms.
        with Pool(processes=workers) as pool:
            results = pool.map(
                _compute_histograms_worker,
                args,
                chunksize=max(1, len(args) // workers),
            )

    return dict(results)
