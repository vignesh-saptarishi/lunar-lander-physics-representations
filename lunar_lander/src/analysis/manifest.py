"""Analysis manifest loader for seed aggregation and cross-config comparison.

Manifests are YAML files that explicitly define what to analyze — which configs,
which seeds, where the runs live. They live in the code repo at
lunar_lander/analysis-manifests/ so analysis is reproducible and version-controlled.

Two manifest types:
  - Seed-agg manifests: flat 'configs' dict, loaded by load_analysis_manifest()
  - Comparison manifests: 'comparisons' dict with named groups, loaded by
    load_comparison_manifest()

Resolution follows the same pattern as training_config.py:
  - File paths (.yaml/.yml) -> loaded directly
  - Builtin names -> resolved from lunar_lander/analysis-manifests/

Usage:
    manifest = load_analysis_manifest("seed-agg/parametric-vs-behavioral")
    manifest = load_comparison_manifest("comparison/parametric-vs-behavioral")
"""

from __future__ import annotations

from pathlib import Path

import yaml


# Builtin manifest directory — parallel to configs/ and batches/.
_MANIFESTS_DIR = Path(__file__).parent.parent.parent / "analysis-manifests"

# Required top-level fields for each manifest type.
_SEED_AGG_REQUIRED_FIELDS = ["experiment", "configs"]
_COMPARISON_REQUIRED_FIELDS = ["experiment", "comparisons"]


def load_analysis_manifest(name_or_path: str) -> dict:
    """Load a seed-aggregation manifest from YAML.

    Resolves the name to a file path (builtin or absolute), reads the YAML,
    validates required fields, and enriches each config with resolved
    seed_dirs (the actual directories to read for each seed).

    Args:
        name_or_path: Builtin name (e.g., "seed-agg/parametric-vs-behavioral")
            or file path to a manifest YAML.

    Returns:
        Parsed manifest dict with configs enriched with 'seed_dirs' lists.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = resolve_manifest_path(name_or_path)

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Validate required fields.
    for field in _SEED_AGG_REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(
                f"Manifest missing required field '{field}'. "
                f"Required: {_SEED_AGG_REQUIRED_FIELDS}. File: {path}"
            )

    # Enrich each config with resolved seed directories.
    for config_name, config in data["configs"].items():
        seed_base = config.get("seed_base", "")
        seeds = config.get("seeds", [])
        config["seed_dirs"] = [f"{seed_base}/s{seed}" for seed in seeds]

    return data


def load_comparison_manifest(name_or_path: str) -> dict:
    """Load a comparison manifest from YAML.

    Comparison manifests have a different schema from seed-agg manifests:
    instead of a flat 'configs' dict, they have 'comparisons' containing
    named groups, each with its own 'configs' sub-dict keyed by variant name.

    Args:
        name_or_path: Builtin name (e.g., "comparison/parametric-vs-behavioral")
            or file path to a manifest YAML.

    Returns:
        Parsed manifest dict with each config enriched with 'seed_dirs' lists.

    Raises:
        FileNotFoundError: If the manifest file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = resolve_manifest_path(name_or_path)

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    # Validate required fields.
    for field in _COMPARISON_REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(
                f"Manifest missing required field '{field}'. "
                f"Required: {_COMPARISON_REQUIRED_FIELDS}. File: {path}"
            )

    # Enrich each config within each comparison with resolved seed directories.
    for comp_name, comp_data in data["comparisons"].items():
        for config_name, config in comp_data.get("configs", {}).items():
            seed_base = config.get("seed_base", "")
            seeds = config.get("seeds", [])
            config["seed_dirs"] = [f"{seed_base}/s{seed}" for seed in seeds]

    return data


def resolve_manifest_path(name_or_path: str) -> Path:
    """Resolve a manifest name or path to a YAML file.

    Public so both seed-agg and comparison loaders can share resolution,
    and external code can resolve paths without loading.

    Same resolution pattern as training_config._resolve_path():
      1. Absolute path or .yaml/.yml extension -> treat as file path.
      2. Otherwise -> try as builtin name in analysis-manifests/ dir.
      3. Neither exists -> FileNotFoundError with available builtins.
    """
    p = Path(name_or_path)

    # Direct file path.
    if p.is_absolute() or p.suffix in (".yaml", ".yml"):
        if p.exists():
            return p
        raise FileNotFoundError(f"Manifest file not found: {name_or_path}")

    # Builtin name resolution.
    builtin_path = _MANIFESTS_DIR / f"{name_or_path}.yaml"
    if builtin_path.exists():
        return builtin_path

    # List available builtins for a helpful error.
    available = (
        [
            str(p.relative_to(_MANIFESTS_DIR).with_suffix(""))
            for p in sorted(_MANIFESTS_DIR.rglob("*.yaml"))
            if not p.stem.startswith("_")
        ]
        if _MANIFESTS_DIR.exists()
        else []
    )
    raise FileNotFoundError(
        f"No analysis manifest '{name_or_path}'. " f"Available: {available}"
    )
