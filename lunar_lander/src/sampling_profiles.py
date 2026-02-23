"""YAML-based sampling profiles for Lunar Lander domain randomization.

A sampling profile narrows or overrides per-parameter sampling ranges,
controlling what portion of the physics space gets sampled. This enables:

  - Curriculum learning: start with easy physics (high TWR, no wind), then
    progressively widen the distribution as the agent improves.
  - Targeted scenarios: isolate one difficulty axis (e.g., wind handling
    with comfortable thrust) for diagnostic evaluation.

Profiles are YAML files where each key is either:
  - A physics parameter name with a scalar (fixed) or [min, max] (range)
  - 'twr_min' / 'twr_max' for compound TWR constraints (rejection-sampled)

Omitted parameters fall back to the full LunarLanderPhysicsConfig.RANGES.

See lunar-lander-testbed.md Section 2.4 for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig

# Directory containing builtin profile YAML files.
# Resolved relative to this source file: lunar_lander/src/ -> lunar_lander/profiles/
_PROFILES_DIR = Path(__file__).parent.parent / "profiles"

# Special keys that aren't physics parameter names — they control
# compound constraints (TWR depends on gravity, thrust, and density jointly).
_SPECIAL_KEYS = {"twr_min", "twr_max"}


@dataclass
class SamplingProfile:
    """A named region of the physics parameter space.

    Attributes:
        overrides: Dict mapping parameter names to either a scalar (float)
            for fixed values or a (min, max) tuple for uniform sub-ranges.
            Parameters not in overrides use full RANGES from PhysicsConfig.
        twr_range: Optional (min_twr, max_twr) constraint. Configs with TWR
            outside this range are rejected and re-sampled.
        name: Optional human-readable name (e.g., "easy", "windy-stable").
    """

    overrides: dict[str, float | tuple[float, float]] = field(default_factory=dict)
    twr_range: tuple[float, float] | None = None
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any], name: str = "") -> SamplingProfile:
        """Build a SamplingProfile from a parsed YAML dict.

        Each key is either a physics parameter name (scalar or [min, max])
        or a special key (twr_min, twr_max). Unknown keys raise ValueError.

        Validation:
          - Parameter names must be in LunarLanderPhysicsConfig.PARAM_NAMES
          - Scalar values must be within the parameter's full RANGES
          - Range [min, max] bounds must be within the parameter's full RANGES
          - min must be <= max for range overrides

        Args:
            d: Dict parsed from YAML. Keys are param names or special keys.
            name: Optional profile name for display purposes.

        Returns:
            A validated SamplingProfile.

        Raises:
            ValueError: If a key is unknown, or a value is out of range.
        """
        overrides = {}
        twr_min = None
        twr_max = None

        for key, value in d.items():
            # Handle special TWR keys
            if key == "twr_min":
                twr_min = float(value)
                continue
            if key == "twr_max":
                twr_max = float(value)
                continue

            # Must be a physics parameter name
            if key not in LunarLanderPhysicsConfig.PARAM_NAMES:
                raise ValueError(
                    f"Unknown key '{key}' in sampling profile. "
                    f"Valid parameter names: {LunarLanderPhysicsConfig.PARAM_NAMES}. "
                    f"Valid special keys: {sorted(_SPECIAL_KEYS)}."
                )

            lo_full, hi_full = LunarLanderPhysicsConfig.RANGES[key]

            if isinstance(value, list):
                # Range override: [min, max] -> (min, max) tuple
                if len(value) != 2:
                    raise ValueError(
                        f"Range for '{key}' must be [min, max], got {value}"
                    )
                lo, hi = float(value[0]), float(value[1])
                if lo > hi:
                    raise ValueError(f"Range for '{key}': min ({lo}) > max ({hi})")
                if lo < lo_full or hi > hi_full:
                    raise ValueError(
                        f"{key} range [{lo}, {hi}] exceeds full range "
                        f"[{lo_full}, {hi_full}]"
                    )
                overrides[key] = (lo, hi)
            else:
                # Scalar override: fixed value
                val = float(value)
                if not (lo_full <= val <= hi_full):
                    raise ValueError(
                        f"{key}={val} outside valid range [{lo_full}, {hi_full}]"
                    )
                overrides[key] = val

        # Build TWR range from optional twr_min / twr_max
        twr_range = None
        if twr_min is not None or twr_max is not None:
            twr_range = (
                twr_min if twr_min is not None else 0.0,
                twr_max if twr_max is not None else float("inf"),
            )

        return cls(overrides=overrides, twr_range=twr_range, name=name)

    @classmethod
    def load(cls, name_or_path: str) -> SamplingProfile:
        """Load a profile by builtin name or file path.

        Resolution order:
          1. Absolute path or .yaml/.yml extension → treat as file path.
          2. Otherwise → try as builtin name relative to profiles dir.
             Supports subdirectories if profiles are organized that way.
          3. Neither exists → raise FileNotFoundError with available names.

        Args:
            name_or_path: Either a builtin name ("easy", "medium", etc.)
                or a path to a YAML file.

        Returns:
            A SamplingProfile loaded from the YAML file.

        Raises:
            FileNotFoundError: If the file or builtin name doesn't exist.
        """
        path = Path(name_or_path)

        # Absolute path or explicit YAML file → use directly
        if path.is_absolute() or path.suffix in (".yaml", ".yml"):
            if not path.exists():
                raise FileNotFoundError(f"Profile file not found: {path}")
            name = path.stem
        else:
            # Try as builtin name (flat or with subdirectory)
            path = _PROFILES_DIR / f"{name_or_path}.yaml"
            if not path.exists():
                available = [
                    str(p.relative_to(_PROFILES_DIR).with_suffix(""))
                    for p in sorted(_PROFILES_DIR.rglob("*.yaml"))
                    if not p.stem.startswith("_")
                ]
                raise FileNotFoundError(
                    f"No builtin profile '{name_or_path}'. " f"Available: {available}"
                )
            name = name_or_path

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data, name=name)

    def sample(
        self,
        rng: np.random.Generator | None = None,
        max_attempts: int = 1000,
    ) -> LunarLanderPhysicsConfig:
        """Sample a physics config from this profile's parameter space.

        For each of the 7 physics parameters:
          - If the profile specifies a scalar, use it directly.
          - If the profile specifies a [min, max] range, draw uniformly.
          - If unspecified, draw from the full RANGES.

        If twr_range is set, rejection-sample until the config's TWR
        falls within the range (same approach as PhysicsConfig.randomize).

        Args:
            rng: NumPy random generator. If None, creates an unseeded one.
            max_attempts: Max rejection-sampling attempts before error.

        Returns:
            A new LunarLanderPhysicsConfig satisfying the profile constraints.

        Raises:
            RuntimeError: If no valid config found within max_attempts.
        """
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(max_attempts):
            kwargs = {}
            for name in LunarLanderPhysicsConfig.PARAM_NAMES:
                if name in self.overrides:
                    override = self.overrides[name]
                    if isinstance(override, tuple):
                        # Range override — sample uniformly within sub-range
                        lo, hi = override
                        kwargs[name] = float(rng.uniform(lo, hi))
                    else:
                        # Scalar override — fixed value
                        kwargs[name] = float(override)
                else:
                    # No override — sample from full range
                    lo, hi = LunarLanderPhysicsConfig.RANGES[name]
                    kwargs[name] = float(rng.uniform(lo, hi))

            config = LunarLanderPhysicsConfig(**kwargs)

            # Check TWR constraint (if any)
            if self.twr_range is None:
                return config

            lo_twr, hi_twr = self.twr_range
            if lo_twr <= config.twr() <= hi_twr:
                return config

        raise RuntimeError(
            f"Could not find config satisfying profile '{self.name}' "
            f"with TWR in {self.twr_range} after {max_attempts} attempts. "
            f"The profile constraints may be too narrow."
        )

    def describe(self) -> str:
        """Human-readable summary of this profile's constraints.

        Returns a multi-line string showing which parameters are overridden
        and what the TWR constraint is (if any). Useful for logging at the
        start of training runs.
        """
        lines = []
        header = f"SamplingProfile '{self.name}'" if self.name else "SamplingProfile"
        lines.append(header)

        if not self.overrides and self.twr_range is None:
            lines.append("  (no overrides — full parameter ranges)")
            return "\n".join(lines)

        for name in LunarLanderPhysicsConfig.PARAM_NAMES:
            full_lo, full_hi = LunarLanderPhysicsConfig.RANGES[name]
            if name in self.overrides:
                override = self.overrides[name]
                if isinstance(override, tuple):
                    lo, hi = override
                    lines.append(
                        f"  {name}: [{lo}, {hi}]  (full: [{full_lo}, {full_hi}])"
                    )
                else:
                    lines.append(f"  {name}: {override}  (fixed)")
            else:
                lines.append(f"  {name}: [{full_lo}, {full_hi}]  (full range)")

        if self.twr_range is not None:
            lo_twr, hi_twr = self.twr_range
            if hi_twr == float("inf"):
                lines.append(f"  TWR constraint: >= {lo_twr}")
            else:
                lines.append(f"  TWR constraint: [{lo_twr}, {hi_twr}]")

        return "\n".join(lines)


@dataclass
class CurriculumSchedule:
    """Step-threshold-based profile progression for RL training.

    Defines a sequence of (step_threshold, profile_name) stages. During
    training, a CurriculumCallback checks the current timestep against the
    schedule and swaps the DomainRandomizationWrapper's profile when a
    threshold is crossed.

    The schedule is typically specified as a CLI string:
        "easy:0,medium:500K,hard:1.5M,full:2.5M"

    Step suffixes: K = x1,000, M = x1,000,000 (case-insensitive).

    Attributes:
        stages: List of (step_threshold, profile_name) tuples, sorted by
            ascending threshold. First threshold must be 0.
    """

    stages: list[tuple[int, str]]

    @classmethod
    def from_string(cls, s: str) -> CurriculumSchedule:
        """Parse a schedule from CLI format: 'profile:step,profile:step,...'.

        Examples:
            "easy:0,medium:500K,hard:1.5M,full:2.5M"
            "easy:0,hard:1000000"

        Step values support K (x1000) and M (x1000000) suffixes.

        Validation:
          - At least one stage required
          - First threshold must be 0
          - Thresholds must be strictly ascending
          - All profile names must be loadable via SamplingProfile.load()

        Args:
            s: Comma-separated "profile:step" pairs.

        Returns:
            A validated CurriculumSchedule.

        Raises:
            ValueError: If format is invalid or thresholds aren't ascending.
            FileNotFoundError: If a profile name can't be loaded.
        """
        stages = []
        for part in s.split(","):
            part = part.strip()
            if ":" not in part:
                raise ValueError(
                    f"Invalid schedule entry '{part}'. Expected 'profile:step'."
                )
            name, step_str = part.rsplit(":", 1)
            name = name.strip()
            step_str = step_str.strip().upper()

            # Parse step with K/M suffix support
            multiplier = 1
            if step_str.endswith("K"):
                multiplier = 1_000
                step_str = step_str[:-1]
            elif step_str.endswith("M"):
                multiplier = 1_000_000
                step_str = step_str[:-1]

            try:
                step = int(float(step_str) * multiplier)
            except ValueError:
                raise ValueError(
                    f"Invalid step value in '{part}'. "
                    f"Expected integer or float with K/M suffix."
                )

            stages.append((step, name))

        if not stages:
            raise ValueError("Curriculum schedule must have at least one stage.")

        # Validate: first threshold must be 0
        if stages[0][0] != 0:
            raise ValueError(
                f"Curriculum schedule must start at 0, "
                f"got first threshold {stages[0][0]}."
            )

        # Validate: thresholds must be strictly ascending
        for i in range(1, len(stages)):
            if stages[i][0] <= stages[i - 1][0]:
                raise ValueError(
                    f"Curriculum thresholds must be strictly ascending. "
                    f"Stage {i} ({stages[i][0]}) <= stage {i-1} ({stages[i-1][0]})."
                )

        # Validate: all profile names must be loadable
        for _, name in stages:
            SamplingProfile.load(name)  # raises FileNotFoundError if missing

        return cls(stages=stages)

    def get_active_profile(self, step: int) -> str:
        """Return the profile name that should be active at the given step.

        Walks the schedule backwards to find the last stage whose threshold
        is <= the current step.

        Args:
            step: Current training timestep.

        Returns:
            Profile name string for the active stage.
        """
        for threshold, name in reversed(self.stages):
            if step >= threshold:
                return name
        # Fallback (shouldn't happen if schedule starts at 0)
        return self.stages[0][1]

    @property
    def initial_profile(self) -> str:
        """The first profile in the schedule (step 0)."""
        return self.stages[0][1]

    @property
    def profile_names(self) -> list[str]:
        """All unique profile names in schedule order."""
        return [name for _, name in self.stages]

    def describe(self) -> str:
        """Human-readable summary for logging at training start."""
        lines = ["Curriculum schedule:"]
        for i, (threshold, name) in enumerate(self.stages):
            if i < len(self.stages) - 1:
                next_threshold = self.stages[i + 1][0]
                duration = next_threshold - threshold
                lines.append(
                    f"  {threshold:>10,} steps: '{name}' "
                    f"({duration:,} steps in this stage)"
                )
            else:
                lines.append(
                    f"  {threshold:>10,} steps: '{name}' (until training ends)"
                )
        return "\n".join(lines)
