"""Physics configuration for the Parameterized Lunar Lander environment.

Defines the 7 continuous physics parameters that can be varied per episode.
Each parameter has a default value matching Gymnasium's LunarLander-v3 and
a valid range for randomization and validation.

The 7 parameters control:
  - Gravitational field strength (gravity)
  - Main and side engine thrust scaling (main_engine_power, side_engine_power)
  - Lander body mass via density (lander_density)
  - Rotational stability (angular_damping)
  - Environmental disturbances (wind_power, turbulence_power)

Together these span a rich space of "physics worlds" — from easy (weak gravity,
strong thrust, no wind) to impossible (strong gravity, weak thrust, heavy
lander, high wind). Calibration validates each config's physical coherence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class LunarLanderPhysicsConfig:
    """7 continuous physics parameters for the Lunar Lander environment.

    Default values reproduce standard Gymnasium LunarLander-v3 behavior exactly.
    Ranges are initial estimates — calibration validates that extremes produce
    meaningfully different but physically coherent behavior.

    Attributes:
        gravity: Gravitational acceleration (negative = downward). Default -10.0
            matches Gymnasium. Range [-12.0, -2.0]. Stronger (more negative)
            gravity makes landing harder; weaker gravity is floatier.

        main_engine_power: Scaling factor for main (upward) thrust impulse.
            Default 13.0 from Gymnasium's MAIN_ENGINE_POWER constant.
            Range [5.0, 25.0]. The actual force applied is:
            impulse = main_engine_power * throttle (where throttle in 0.5..1.0).
            Lower values mean the lander can't fight gravity as effectively.

        side_engine_power: Scaling factor for lateral thrust impulse.
            Default 0.6 from Gymnasium's SIDE_ENGINE_POWER constant.
            Range [0.2, 1.5]. Affects angular correction ability — too low
            means the lander can't correct its orientation fast enough.

        lander_density: Box2D fixture density for the lander body polygon.
            Default 5.0 (from Gymnasium source line 384). Box2D computes
            mass = density * polygon_area, so higher density = heavier lander.
            Range [2.5, 10.0]. Leg density stays fixed at 1.0 — varying it
            would conflate structural mass with payload mass.

        angular_damping: Box2D body angular damping coefficient.
            Default 0.0 (Box2D default — free rotation, not set in Gymnasium).
            Range [0.0, 5.0]. Box2D applies this each step as:
            angular_velocity *= 1 / (1 + dt * damping).
            0 = free spin (default behavior), higher = more rotationally stable.
            High values make the lander sluggish but resistant to tumbling.

        wind_power: Maximum magnitude of horizontal wind force.
            Default 15.0 from Gymnasium's wind_power parameter.
            Range [0.0, 30.0]. Wind is always "enabled" in our fork (no
            enable_wind bool). Set to 0.0 for no wind — simpler parameterization.
            Wind pattern is quasi-periodic: tanh(sin(2k*t) + sin(pi*k*t)).

        turbulence_power: Maximum magnitude of angular torque from turbulence.
            Default 1.5 from Gymnasium's turbulence_power parameter.
            Range [0.0, 5.0]. Applied as random torque each step, disturbing
            lander orientation. Independent of wind_power.
    """

    gravity: float = -10.0
    main_engine_power: float = 13.0
    side_engine_power: float = 0.6
    lander_density: float = 5.0
    angular_damping: float = 0.0
    wind_power: float = 15.0
    turbulence_power: float = 1.5

    # --- Class-level constants ---

    # Valid ranges for each parameter: {name: (min, max)}.
    # Used by __post_init__ for validation and randomize() for sampling.
    RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "gravity": (-12.0, -2.0),
        "main_engine_power": (5.0, 25.0),
        "side_engine_power": (0.2, 1.5),
        "lander_density": (2.5, 10.0),
        "angular_damping": (0.0, 5.0),
        "wind_power": (0.0, 30.0),
        "turbulence_power": (0.0, 5.0),
    }

    # Lander body polygon area in Box2D world coordinates.
    # Computed from LANDER_POLY = [(-14,17),(-17,0),(-17,-10),(17,-10),(17,0),(14,17)]
    # scaled by SCALE=30. Shoelace formula gives 867 px² / 30² = 0.96333 world².
    # This is a constant — only changes if the lander shape changes.
    # Used to compute mass (= density × area) and thrust-to-weight ratio.
    BODY_AREA: ClassVar[float] = 867.0 / 900.0

    # Ordered parameter names — defines the canonical order for observation
    # vectors, serialization, and array conversion. Matches field declaration order.
    PARAM_NAMES: ClassVar[list[str]] = [
        "gravity",
        "main_engine_power",
        "side_engine_power",
        "lander_density",
        "angular_damping",
        "wind_power",
        "turbulence_power",
    ]

    def __post_init__(self):
        """Validate all parameters are within their defined ranges.

        Raises ValueError with a descriptive message if any parameter
        is out of range. This catches configuration errors early rather
        than producing mysterious Box2D behavior at simulation time.
        """
        for name in self.PARAM_NAMES:
            value = getattr(self, name)
            lo, hi = self.RANGES[name]
            if not (lo <= value <= hi):
                raise ValueError(f"{name}={value} is out of range [{lo}, {hi}]")

    def twr(self, fps: int = 50) -> float:
        """Thrust-to-weight ratio at full throttle.

        In Box2D, each step the engine applies an impulse of
        main_engine_power * m_power (where m_power=1.0 at full throttle).
        Gravity applies an impulse of mass * |gravity| / fps per step.
        TWR = engine_impulse / gravity_impulse.

        TWR > 1 means the lander can hover at full throttle.
        TWR of 5-10 means comfortable maneuvering margin.
        TWR of 2-3 means the lander must use most of its thrust just to hover.

        Args:
            fps: Simulation framerate (default 50, matching Box2D step rate).
        """
        mass = self.lander_density * self.BODY_AREA
        return self.main_engine_power * fps / (mass * abs(self.gravity))

    @classmethod
    def randomize(
        cls,
        rng: np.random.Generator | None = None,
        twr_range: tuple[float, float] | None = None,
        max_attempts: int = 1000,
    ) -> LunarLanderPhysicsConfig:
        """Create a config with uniform random values within each parameter's range.

        Args:
            rng: NumPy random generator for reproducibility. If None, creates
                 a new default (unseeded) generator. For reproducible experiments,
                 pass np.random.default_rng(seed).
            twr_range: Optional (min_twr, max_twr) constraint. If set, rejection-
                 samples until the config's thrust-to-weight ratio falls within
                 this range. Useful for ensuring solvable configs (e.g., twr >= 3)
                 or curriculum training (start easy, widen range).
            max_attempts: Max rejection sampling attempts before raising an error.

        Returns:
            A new LunarLanderPhysicsConfig with random parameter values.

        Raises:
            RuntimeError: If twr_range is set and no valid config is found
                within max_attempts.
        """
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(max_attempts):
            kwargs = {}
            for name in cls.PARAM_NAMES:
                lo, hi = cls.RANGES[name]
                kwargs[name] = float(rng.uniform(lo, hi))
            config = cls(**kwargs)

            if twr_range is None:
                return config

            lo_twr, hi_twr = twr_range
            if lo_twr <= config.twr() <= hi_twr:
                return config

        raise RuntimeError(
            f"Could not find config with TWR in {twr_range} "
            f"after {max_attempts} attempts. The range may be too narrow "
            f"for the current parameter ranges."
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage.

        Returns parameters in canonical PARAM_NAMES order. Values are plain
        Python floats (not numpy) for JSON compatibility.
        """
        return {name: getattr(self, name) for name in self.PARAM_NAMES}

    @classmethod
    def from_dict(cls, d: dict) -> LunarLanderPhysicsConfig:
        """Deserialize from a dict (e.g., loaded from JSON).

        Only reads known parameter names — ignores extra keys for
        forward compatibility (e.g., if future versions add parameters,
        old configs can still be loaded without error).
        """
        kwargs = {name: d[name] for name in cls.PARAM_NAMES if name in d}
        return cls(**kwargs)

    def config_hash(self) -> str:
        """Deterministic hash for calibration caching.

        Rounds floats to 6 decimal places to avoid floating-point noise
        causing cache misses for effectively identical configs. Returns
        first 16 hex chars of SHA-256 (64 bits — sufficient for caching,
        not for security).
        """
        # Round to 6 decimals to absorb float arithmetic noise.
        # Sort keys for determinism across Python versions.
        rounded = {name: round(getattr(self, name), 6) for name in self.PARAM_NAMES}
        canonical = json.dumps(rounded, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def as_array(self) -> np.ndarray:
        """Return physics params as a float32 numpy array in canonical order.

        Used for appending to the observation vector. Order matches PARAM_NAMES:
        [gravity, main_engine_power, side_engine_power, lander_density,
         angular_damping, wind_power, turbulence_power].

        Values are raw (unnormalized) — normalization happens at training time
        via observation wrappers, not at collection time.
        """
        return np.array(
            [getattr(self, name) for name in self.PARAM_NAMES],
            dtype=np.float32,
        )
