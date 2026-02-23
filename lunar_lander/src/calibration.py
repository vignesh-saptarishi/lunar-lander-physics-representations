"""Calibration system for the parameterized Lunar Lander environment.

Runs 5 canonical maneuvers on a fresh env instance and compares simulated
outcomes against analytical Newtonian predictions. This validates that:
  1. Physics parameters are wired correctly (gravity, thrust, mass, etc.)
  2. Box2D simulation matches expected physics within discretization tolerance
  3. Different configs produce quantitatively different behavior

The first 4 maneuvers have analytical ground truth (Newtonian kinematics).
The 5th (hover attempt) is purely behavioral — measures controllability.

Measurement strategy:
  We do NOT teleport the lander body after reset(). Teleporting creates joint
  constraint violations (legs are attached via revolute joints) that corrupt
  measurements with large corrective forces. Instead we:
    - Free fall / thrust: measure velocity CHANGE (Δv) from the natural
      post-reset state. The change is independent of initial conditions.
    - Angular decay: set angularVelocity only (no position teleport), let
      1 step settle joint transients, then measure decay over 10 steps.

Tolerance rationale:
  Box2D uses discrete timesteps (dt = 1/FPS = 0.02s) with velocity/position
  iterations. Engine dispersion noise and angle drift add ~3-5% error.
  Tolerances: 5% for gravity (clean), 15% for thrust (dispersion + angle),
  10% for angular decay (joint motor torque offset).

Coordinate system:
  All measurements use Box2D world coordinates (body.position, body.linearVelocity).
  These are in "meters" (actually VIEWPORT/SCALE units). We do NOT use the
  normalized state vector coordinates (which are divided by viewport factors).

Usage:
    from lunar_lander.src.calibration import calibrate, get_or_calibrate
    from lunar_lander.src.physics_config import LunarLanderPhysicsConfig

    config = LunarLanderPhysicsConfig(gravity=-5.0)
    result = calibrate(config)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig
from lunar_lander.src.env import (
    ParameterizedLunarLander,
    FPS,
    SCALE,
    MAIN_ENGINE_Y_LOCATION,
    SIDE_ENGINE_AWAY,
)


# --- Tolerances for analytical ground truth comparison ---
# These are relative tolerances (e.g., 0.05 = 5%).
#
# Free fall: gravity is the only external force on the COM. Joint forces
# are internal and don't affect COM velocity. Extremely clean measurement.
GRAVITY_TOL = 0.05  # 5%
#
# Thrust: engine dispersion noise (random nozzle direction per step) and
# angle drift (thrust rotates lander, changing thrust decomposition) add
# ~5-10% error. We use windowed measurement (skip first 5 steps for
# transients, measure 10 steps before angle drifts too far).
THRUST_TOL = 0.15  # 15%
#
# Side thrust: off-center impulse creates torque that rotates lander,
# changing the lateral thrust direction over time.
SIDE_THRUST_TOL = 0.10  # 10%
#
# Angular decay: leg joint motor torques add a constant-ish angular
# acceleration offset (~0.07 rad/s per step). We use high initial omega
# (10 rad/s) so damping dominates, and measure from the settled state
# (1 settle step after override).
ANGULAR_TOL = 0.10  # 10%


@dataclass
class ManeuverResult:
    """Result of a single calibration maneuver.

    Stores measured values from Box2D simulation, analytical predictions
    (if applicable), and whether the measurement is within tolerance.
    """

    name: str
    measured: dict = field(default_factory=dict)
    analytical: dict = field(default_factory=dict)
    passed: bool = True
    notes: str = ""


@dataclass
class CalibrationResult:
    """Complete calibration result for a physics config.

    Contains results from all 5 canonical maneuvers, the physics config
    that was tested, and overall pass/fail status.
    """

    physics_config: LunarLanderPhysicsConfig
    maneuvers: dict[str, ManeuverResult] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """True if all maneuvers with analytical ground truth passed."""
        return all(m.passed for m in self.maneuvers.values())

    def summary(self) -> str:
        """Human-readable summary of calibration results."""
        lines = [f"Calibration for config hash {self.physics_config.config_hash()}:"]
        for name, result in self.maneuvers.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"  [{status}] {name}")
            for key, val in result.measured.items():
                anal = result.analytical.get(key)
                if anal is not None:
                    lines.append(
                        f"    {key}: measured={val:.4f}, analytical={anal:.4f}"
                    )
                else:
                    lines.append(f"    {key}: {val:.4f}")
            if result.notes:
                lines.append(f"    note: {result.notes}")
        overall = "PASS" if self.all_passed else "FAIL"
        lines.append(f"  Overall: {overall}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage in episode metadata."""
        return {
            "config_hash": self.physics_config.config_hash(),
            "all_passed": self.all_passed,
            "maneuvers": {
                name: {
                    "measured": m.measured,
                    "analytical": m.analytical,
                    "passed": m.passed,
                    "notes": m.notes,
                }
                for name, m in self.maneuvers.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict, config: LunarLanderPhysicsConfig) -> CalibrationResult:
        """Deserialize from dict."""
        result = cls(physics_config=config)
        for name, m_dict in d.get("maneuvers", {}).items():
            result.maneuvers[name] = ManeuverResult(
                name=name,
                measured=m_dict.get("measured", {}),
                analytical=m_dict.get("analytical", {}),
                passed=m_dict.get("passed", True),
                notes=m_dict.get("notes", ""),
            )
        return result


def _make_clean_config(config: LunarLanderPhysicsConfig) -> LunarLanderPhysicsConfig:
    """Create a copy of config with wind/turbulence disabled.

    The first 4 analytical maneuvers need clean conditions (no external
    disturbances) so we can compare against Newtonian predictions. Wind
    and turbulence are only relevant for the hover behavioral test.
    """
    return LunarLanderPhysicsConfig(
        gravity=config.gravity,
        main_engine_power=config.main_engine_power,
        side_engine_power=config.side_engine_power,
        lander_density=config.lander_density,
        angular_damping=config.angular_damping,
        wind_power=0.0,
        turbulence_power=0.0,
    )


def calibrate_free_fall(config: LunarLanderPhysicsConfig) -> ManeuverResult:
    """Maneuver 1: Free fall — measure gravity's effect on vertical velocity.

    Tests that gravity is correctly applied. We measure the CHANGE in
    vertical velocity (Δvy) over N steps with no thrust, no wind.

    Why measure Δvy instead of absolute position:
      The lander starts with a random impulse from reset(), giving it an
      unknown initial velocity. But Δvy depends only on gravity (constant
      acceleration), so it's independent of initial conditions.

    Why no teleportation:
      Teleporting the lander body creates joint constraint violations
      (legs are attached via revolute joints at the spawn position).
      Box2D corrects these with large forces that corrupt measurements.

    Analytical model:
      Δvy = g * Δt  (constant acceleration under gravity)
      For discrete steps: Δvy = g * n_steps * dt  (exact for Euler integration
      with constant acceleration — no discretization error).

    Joint force effect on linear velocity:
      Leg joint forces are internal to the lander+legs system. They don't
      affect the system's center-of-mass velocity. They DO affect the
      individual body velocities (lander vs legs), but the effect is small
      because legs are much lighter than the lander (density 1 vs 5).
      Empirically < 0.1% error.
    """
    env = ParameterizedLunarLander(
        render_mode=None,
        physics_config=_make_clean_config(config),
    )
    env.reset(seed=0)

    dt = 1.0 / FPS
    n_steps = 30  # 0.6s — enough for measurable fall, short enough to stay in bounds
    no_action = np.array([0.0, 0.0], dtype=np.float32)

    # Record initial vertical velocity (includes random impulse effect).
    vy0 = env.lander.linearVelocity.y

    for _ in range(n_steps):
        env.step(no_action)

    vy_final = env.lander.linearVelocity.y
    env.close()

    # Measured and analytical velocity change.
    delta_vy_measured = vy_final - vy0
    delta_vy_analytical = config.gravity * n_steps * dt

    vy_error = abs(delta_vy_measured - delta_vy_analytical) / abs(delta_vy_analytical)
    passed = vy_error < GRAVITY_TOL

    return ManeuverResult(
        name="free_fall",
        measured={
            "delta_vy": float(delta_vy_measured),
            "vy_final": float(vy_final),
        },
        analytical={
            "delta_vy": float(delta_vy_analytical),
        },
        passed=passed,
        notes=f"vy_error={vy_error:.4f}, t={n_steps * dt:.2f}s",
    )


def calibrate_full_thrust(config: LunarLanderPhysicsConfig) -> ManeuverResult:
    """Maneuver 2: Full main engine thrust — measure thrust-to-weight ratio.

    Tests that main_engine_power and lander mass interact correctly.
    Fire main engine at full power (action=[1.0, 0.0]) and measure the
    net vertical acceleration, from which we compute thrust-to-weight ratio.

    Windowed measurement (steps 5-15):
      We skip the first 5 steps to let initial impulse transients settle,
      then measure over 10 steps before the lander's angle drifts too far
      (engine thrust direction depends on lander angle).

    Analytical model:
      Engine impulse per step (upright lander):
        impulse_y = (MAIN_ENGINE_Y_LOCATION / SCALE) * main_engine_power * m_power
      where m_power = (clip(action[0], 0, 1) + 1) * 0.5 = 1.0 for action[0]=1.0

      Engine acceleration = impulse / (mass * dt)
      Net acceleration = engine_accel + gravity  (gravity is negative)
      TWR = engine_accel / |gravity|

    Error sources (~5-10%):
      - Engine nozzle dispersion: random per step, partially averages out
      - Angle drift: lander rotates from off-center thrust, changing
        the vertical component of thrust (cos(angle) factor)
      - The MAIN_ENGINE_Y_LOCATION offset means thrust isn't at COM
    """
    env = ParameterizedLunarLander(
        render_mode=None,
        physics_config=_make_clean_config(config),
    )
    env.reset(seed=0)
    mass = env.lander.mass

    dt = 1.0 / FPS
    full_thrust = np.array([1.0, 0.0], dtype=np.float32)

    # Run 20 steps, recording vy at each step for windowed analysis.
    vys = [env.lander.linearVelocity.y]
    for _ in range(20):
        env.step(full_thrust)
        vys.append(env.lander.linearVelocity.y)
    env.close()

    # Windowed measurement: steps 5 to 15 (skip transients, avoid angle drift).
    window_start, window_end = 5, 15
    n_window = window_end - window_start
    delta_vy = vys[window_end] - vys[window_start]
    t_window = n_window * dt

    # Net acceleration from velocity change.
    a_net_measured = delta_vy / t_window
    # Engine acceleration = net - gravity (gravity is negative, so engine = net + |g|)
    a_engine_measured = a_net_measured - config.gravity
    twr_measured = a_engine_measured / abs(config.gravity)

    # Analytical TWR.
    impulse_factor = MAIN_ENGINE_Y_LOCATION / SCALE  # ≈ 0.133
    engine_accel_analytical = (impulse_factor * config.main_engine_power) / (mass * dt)
    twr_analytical = engine_accel_analytical / abs(config.gravity)

    twr_error = abs(twr_measured - twr_analytical) / max(abs(twr_analytical), 0.01)
    passed = twr_error < THRUST_TOL

    return ManeuverResult(
        name="full_thrust",
        measured={
            "a_engine": float(a_engine_measured),
            "twr": float(twr_measured),
            "mass": float(mass),
        },
        analytical={
            "a_engine": float(engine_accel_analytical),
            "twr": float(twr_analytical),
        },
        passed=passed,
        notes=f"twr_error={twr_error:.4f}, mass={mass:.3f}kg",
    )


def calibrate_side_thrust(config: LunarLanderPhysicsConfig) -> ManeuverResult:
    """Maneuver 3: Full side engine thrust — measure lateral acceleration.

    Tests that side_engine_power is correctly applied. Fire the side engine
    at full power (action=[0.0, 1.0]) and measure lateral velocity change.

    Windowed measurement (steps 5-15):
      Same rationale as full thrust — skip transients, measure before
      angle drift distorts the lateral acceleration direction.

    Side engine fires when |action[1]| > 0.5. For action[1]=1.0:
      s_power = clip(|1.0|, 0.5, 1.0) = 1.0
      direction = sign(1.0) = +1

    Analytical model (upright lander):
      ox ≈ side[0] * direction * SIDE_ENGINE_AWAY / SCALE
      For angle=0: side = (-cos(0), sin(0)) = (-1, 0)
      ox = -1 * 1 * SIDE_ENGINE_AWAY / SCALE = -0.4
      Impulse to lander = -ox * side_engine_power * s_power = 0.4 * SEP
      Acceleration = impulse / (mass * dt)

    The off-center impulse also creates torque, which rotates the lander.
    Over many steps this distorts the lateral acceleration measurement.
    """
    env = ParameterizedLunarLander(
        render_mode=None,
        physics_config=_make_clean_config(config),
    )
    env.reset(seed=0)
    mass = env.lander.mass

    dt = 1.0 / FPS
    side_action = np.array([0.0, 1.0], dtype=np.float32)

    # Run 20 steps, recording vx.
    vxs = [env.lander.linearVelocity.x]
    for _ in range(20):
        env.step(side_action)
        vxs.append(env.lander.linearVelocity.x)
    env.close()

    # Windowed measurement: steps 5 to 15.
    window_start, window_end = 5, 15
    n_window = window_end - window_start
    delta_vx = vxs[window_end] - vxs[window_start]
    t_window = n_window * dt

    a_lateral_measured = delta_vx / t_window

    # Analytical lateral acceleration.
    impulse_factor = SIDE_ENGINE_AWAY / SCALE  # 12/30 = 0.4
    a_lateral_analytical = (impulse_factor * config.side_engine_power) / (mass * dt)

    # Compare absolute values (side engine direction depends on lander angle).
    a_error = abs(abs(a_lateral_measured) - a_lateral_analytical) / max(
        a_lateral_analytical, 0.01
    )
    passed = a_error < SIDE_THRUST_TOL

    return ManeuverResult(
        name="side_thrust",
        measured={
            "a_lateral": float(a_lateral_measured),
        },
        analytical={
            "a_lateral": float(a_lateral_analytical),
        },
        passed=passed,
        notes=f"a_error={a_error:.4f}",
    )


def calibrate_angular_decay(config: LunarLanderPhysicsConfig) -> ManeuverResult:
    """Maneuver 4: Angular velocity decay under damping.

    Tests that angular_damping is correctly applied. Set a high initial
    angular velocity and measure the decay over 10 steps.

    Strategy — no position teleport, only angularVelocity override:
      Teleporting the lander's position violates leg joint constraints,
      causing large corrective forces that dominate angular dynamics.
      Setting angularVelocity alone doesn't violate position constraints,
      so joint corrections are minimal.

      We let 1 step pass after setting angularVelocity ("settle step")
      to let the joint equilibrate to the new angular state, then measure
      from the settled omega over 10 steps.

    Residual error source — leg joint motor torques:
      The leg joints have motors (LEG_SPRING_TORQUE=40) that create a
      constant-ish angular acceleration offset of ~0.07 rad/s per step.
      Using high initial omega (10 rad/s) ensures damping effects dominate
      this offset. Over 10 steps, the joint torque adds ~0.7 rad/s while
      damping removes a fraction of omega — the damping signal is larger.

    Box2D angular damping model:
      Per step: ω *= 1 / (1 + dt * damping)
      After N steps: ω_N = ω_0 * (1 / (1 + dt * damping))^N
      For damping=0: ω stays approximately constant.
      For damping=5, dt=0.02: per-step factor = 1/1.1 = 0.909
    """
    env = ParameterizedLunarLander(
        render_mode=None,
        physics_config=_make_clean_config(config),
    )
    env.reset(seed=0)

    dt = 1.0 / FPS
    no_action = np.array([0.0, 0.0], dtype=np.float32)

    # Set high angular velocity (no position teleport — joints stay valid).
    env.lander.angularVelocity = 10.0

    # Settle step: let joints equilibrate to the new angular state.
    # The first step after override may have transient joint corrections.
    env.step(no_action)
    omega_start = env.lander.angularVelocity

    # Measure over 10 steps.
    n_measure = 10
    for _ in range(n_measure):
        env.step(no_action)
    omega_end = env.lander.angularVelocity
    env.close()

    # Analytical prediction from the settled omega_start.
    damping = config.angular_damping
    decay_factor = (1.0 / (1.0 + dt * damping)) ** n_measure
    omega_analytical = omega_start * decay_factor

    if abs(omega_analytical) > 0.01:
        omega_error = abs(omega_end - omega_analytical) / abs(omega_analytical)
    else:
        # When analytical prediction is near zero, use absolute comparison.
        omega_error = abs(omega_end - omega_analytical)

    passed = omega_error < ANGULAR_TOL

    return ManeuverResult(
        name="angular_decay",
        measured={
            "omega_start": float(omega_start),
            "omega_end": float(omega_end),
            "decay_ratio": float(omega_end / omega_start) if omega_start != 0 else 0.0,
        },
        analytical={
            "omega_end": float(omega_analytical),
            "decay_factor": float(decay_factor),
        },
        passed=passed,
        notes=f"omega_error={omega_error:.4f}, damping={damping}",
    )


def calibrate_hover(
    config: LunarLanderPhysicsConfig,
    heuristic_fn: Callable,
) -> ManeuverResult:
    """Maneuver 5: Hover attempt with heuristic controller.

    Purely behavioral — no analytical ground truth. Measures whether the
    heuristic controller can maintain altitude under this physics config.

    This reveals controllability: configs where thrust-to-weight < 1 will
    crash, configs with strong wind will drift, etc. The heuristic is tuned
    for default physics, so degradation under varied physics is expected and
    informative.

    Uses the env's own observation (not manually constructed) to avoid
    subtle coordinate mismatches.

    Metrics:
        - mean_altitude: average y position (higher = floating, lower = falling)
        - altitude_std: stability of hover (lower = more stable)
        - fuel_used: total engine activation (sum of |main| + |side| power)
        - angle_std: rotational stability (lower = more stable)
        - steps_survived: how long before termination (landing or crash)
    """
    # Use the actual config (including wind) — this tests real controllability.
    env = ParameterizedLunarLander(render_mode=None, physics_config=config)
    obs, _ = env.reset(seed=42)

    n_steps = 200
    altitudes = []
    angles = []
    fuel = 0.0

    for _ in range(n_steps):
        # Heuristic uses base 8D state (obs[:8]), ignores physics params.
        action = heuristic_fn(obs)
        obs, _, terminated, _, _ = env.step(action)

        altitudes.append(float(env.lander.position.y))
        angles.append(float(env.lander.angle))

        # Track fuel: main engine fires when action[0] > 0, side when |action[1]| > 0.5
        if action[0] > 0:
            fuel += float((np.clip(action[0], 0, 1) + 1) * 0.5)
        if abs(action[1]) > 0.5:
            fuel += float(np.clip(abs(action[1]), 0.5, 1.0))

        if terminated:
            break

    env.close()

    altitudes_arr = np.array(altitudes)
    angles_arr = np.array(angles)

    return ManeuverResult(
        name="hover",
        measured={
            "mean_altitude": float(np.mean(altitudes_arr)),
            "altitude_std": float(np.std(altitudes_arr)),
            "fuel_used": float(fuel),
            "angle_std": float(np.std(angles_arr)),
            "steps_survived": len(altitudes),
        },
        analytical={},  # No analytical ground truth for hover.
        passed=True,  # Hover always "passes" — it's informational.
        notes=f"steps={len(altitudes)}",
    )


def calibrate(
    config: LunarLanderPhysicsConfig,
    heuristic_fn: Callable | None = None,
) -> CalibrationResult:
    """Run all calibration maneuvers for a physics config.

    Runs the 4 analytical maneuvers unconditionally. The hover maneuver
    runs only if a heuristic function is provided.

    Args:
        config: Physics config to calibrate.
        heuristic_fn: Optional heuristic policy function for hover test.
            Signature: (obs: np.ndarray) -> np.ndarray.

    Returns:
        CalibrationResult with all maneuver outcomes.
    """
    result = CalibrationResult(physics_config=config)

    result.maneuvers["free_fall"] = calibrate_free_fall(config)
    result.maneuvers["full_thrust"] = calibrate_full_thrust(config)
    result.maneuvers["side_thrust"] = calibrate_side_thrust(config)
    result.maneuvers["angular_decay"] = calibrate_angular_decay(config)

    if heuristic_fn is not None:
        result.maneuvers["hover"] = calibrate_hover(config, heuristic_fn)

    return result


# --- Calibration cache ---
# Keyed by config hash. Since calibration is deterministic for a given config
# (fixed seed, no randomness beyond engine dispersion which is seeded),
# we can cache results to avoid redundant computation during batch collection.
_cache: dict[str, CalibrationResult] = {}


def get_or_calibrate(
    config: LunarLanderPhysicsConfig,
    heuristic_fn: Callable | None = None,
) -> CalibrationResult:
    """Get cached calibration result or compute new one.

    Uses config_hash() as cache key. Two configs with the same parameter
    values will share a cache entry.

    Args:
        config: Physics config to calibrate.
        heuristic_fn: Optional heuristic for hover test.

    Returns:
        CalibrationResult (possibly cached).
    """
    key = config.config_hash()
    if key not in _cache:
        _cache[key] = calibrate(config, heuristic_fn)
    return _cache[key]


def clear_cache():
    """Clear the calibration cache. Useful for testing."""
    _cache.clear()
