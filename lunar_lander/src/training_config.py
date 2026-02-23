"""Training config YAML loader for Lunar Lander RL training.

Loads a YAML file that fully defines a training run: agent variant, algorithm,
network architecture, physics profile, and output paths. This replaces long
CLI invocations with version-controllable config files.

Resolution order for train_rl.py (highest -> lowest):
  1. CLI args (--variant, --total-steps, etc.)
  2. YAML config (--config)
  3. TRAINING_DEFAULTS (hardcoded below)

CLI overrides are applied in train_rl.py, not here. This module only handles
YAML loading and merging with TRAINING_DEFAULTS.

Two loaders:
  - load_training_config() -- single training run config
  - load_batch_config() -- list of config names for batch execution

Usage:
    config = load_training_config("labeled-ppo")     # builtin name
    config = load_training_config("path/to/custom.yaml")  # file path

    runs = load_batch_config("all-ppo")              # builtin batch
    runs = load_batch_config("path/to/batch.yaml")   # custom batch
"""

from pathlib import Path

import yaml


# Builtin config directories (parallel to profiles/).
# These don't exist yet -- Task 4 creates the actual YAML files.
# The loader resolves builtin names to these directories.
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"
_BATCHES_DIR = Path(__file__).parent.parent / "batches"


# Defaults for all training parameters. Every key here can be overridden
# by a YAML config or CLI arg.
#
# Design principle: structural/hyperparameter defaults (net_arch, n_rays,
# history_k) are fine -- they rarely change. Operational params (variant,
# algo, total_steps, run_dir) are None = REQUIRED. Every YAML config
# must set these explicitly. No silent fallbacks for things that matter.
TRAINING_DEFAULTS = {
    # --- Agent (REQUIRED -- must be in YAML or CLI) ---
    "variant": None,  # REQUIRED: labeled | blind | history
    "algo": None,  # REQUIRED: ppo | sac
    # --- Structural defaults (rarely changed) ---
    "history_k": 8,  # history stack depth (history variant only)
    "n_rays": 7,  # terrain sensing rays
    # --- Operational (REQUIRED -- must be in YAML or CLI) ---
    "total_steps": None,  # REQUIRED: total env steps
    "n_envs": 8,  # sensible default -- hardware-dependent
    "seed": 42,
    # --- Schedule ---
    "eval_freq": 50_000,
    "checkpoint_freq": 100_000,
    "video_freq": 0,  # 0 = disabled
    # --- Physics ---
    "profile": None,  # sampling profile name or path
    "curriculum": None,  # curriculum schedule string
    "twr_min": None,  # legacy TWR constraint
    "twr_max": None,  # legacy TWR constraint
    # --- Early stopping ---
    # Stop training when landed_pct >= threshold for N consecutive evals.
    # None = disabled (train for full total_steps).
    "early_stop_landed_pct": None,  # e.g., 95.0
    "early_stop_patience": 3,  # consecutive evals above threshold
    # --- Algorithm hyperparameters ---
    # Override rl_common defaults. None = use rl_common defaults.
    # These are passed as algo_kwargs to train().
    "ent_coef": None,  # entropy coefficient (PPO default: 0.01, SAC: "auto")
    # --- Network ---
    # None = use DEFAULT_POLICY_KWARGS from rl_common (3x256 pi+vf).
    # List like [64, 64] = shared actor/critic MLP layers.
    # Dict like {"pi": [256,256,256], "vf": [256,256,256]} = separate (PPO only).
    "net_arch": None,
    # --- Output (REQUIRED -- must be in YAML or CLI) ---
    "run_dir": None,  # REQUIRED: single directory for all outputs
}


def _resolve_path(name_or_path: str, builtin_dir: Path, kind: str) -> Path:
    """Resolve a name or path to a YAML file.

    Resolution order:
      1. Absolute path or .yaml/.yml extension → treat as file path directly.
      2. Otherwise → try as builtin name relative to builtin_dir.
         Supports subdirectories (e.g., 'physics-only/labeled-ppo-easy'
         resolves to builtin_dir/physics-only/labeled-ppo-easy.yaml).
      3. Neither exists → raise FileNotFoundError with available builtins.

    Args:
        name_or_path: Builtin name (e.g., "labeled-ppo-easy" or
            "physics-only/labeled-ppo-easy") or file path.
        builtin_dir: Directory containing builtin YAML files.
        kind: Human-readable kind for error messages ("config" or "batch").

    Returns:
        Resolved Path to the YAML file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(name_or_path)

    # Absolute path or explicit YAML file → use directly as file path
    if path.is_absolute() or path.suffix in (".yaml", ".yml"):
        if not path.exists():
            raise FileNotFoundError(f"{kind.title()} file not found: {path}")
        return path

    # Try as builtin name (flat or with subdirectory)
    builtin_path = builtin_dir / f"{name_or_path}.yaml"
    if builtin_path.exists():
        return builtin_path

    # Nothing found — list available builtins for a helpful error
    available = (
        [
            str(p.relative_to(builtin_dir).with_suffix(""))
            for p in sorted(builtin_dir.rglob("*.yaml"))
            if not p.stem.startswith("_")
        ]
        if builtin_dir.exists()
        else []
    )
    raise FileNotFoundError(
        f"No builtin {kind} '{name_or_path}'. " f"Available: {available}"
    )


def load_training_config(name_or_path: str) -> dict:
    """Load a training config from YAML and merge with defaults.

    Starts from TRAINING_DEFAULTS, overlays any keys present in the YAML.
    Unknown keys in the YAML are silently ignored (forward-compatible).

    Args:
        name_or_path: Builtin config name (e.g., "labeled-ppo" resolves to
            lunar_lander/configs/labeled-ppo.yaml) or path to a YAML file.

    Returns:
        Config dict with all TRAINING_DEFAULTS keys populated.

    Raises:
        FileNotFoundError: If the config name or path doesn't exist.
    """
    path = _resolve_path(name_or_path, _CONFIGS_DIR, "config")

    with open(path) as f:
        yaml_data = yaml.safe_load(f) or {}

    # Start from defaults, overlay only recognized YAML values.
    # Unknown keys are silently dropped -- this makes the config format
    # forward-compatible (new keys in YAML won't break old loaders).
    config = TRAINING_DEFAULTS.copy()
    for key, value in yaml_data.items():
        if key in config:
            config[key] = value

    return config


def load_batch_config(name_or_path: str) -> list[str]:
    """Load a batch config: a flat list of training config names.

    Batch configs are simple YAML files with a single "runs" key containing
    a list of training config names. Each name is resolved independently
    by load_training_config() when the batch is executed.

    Args:
        name_or_path: Builtin batch name (e.g., "all-ppo" resolves to
            lunar_lander/batches/all-ppo.yaml) or path to a YAML file.

    Returns:
        List of training config names (strings).

    Raises:
        FileNotFoundError: If the batch name or path doesn't exist.
        ValueError: If the batch has no runs.
    """
    path = _resolve_path(name_or_path, _BATCHES_DIR, "batch")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"Batch config has no runs: {path}")

    return runs
