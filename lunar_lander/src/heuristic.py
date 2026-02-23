"""Heuristic PD controller for Lunar Lander.

Adapted from Gymnasium's built-in controller (gymnasium/envs/box2d/lunar_lander.py,
line ~794). This is a proportional-derivative controller that computes angle and
hover targets from position/velocity errors.

Important: Tuned for DEFAULT physics parameters only (gravity=-10, main_engine_power=13, etc.).
Under varied physics, it degrades naturally — this degradation is the key experimental signal
showing why adaptation to physics is needed.

The controller takes the base 8D state (positions, velocities, angle, leg contacts) and
returns continuous actions. It does NOT see physics parameters — this makes it the
"blind fixed policy" baseline for data collection.
"""

import numpy as np


def heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """PD controller that attempts to land the lander.

    Takes the base state vector (first 8 dims of observation) and computes
    a continuous action. Works with the full 15D obs too (ignores physics params).

    Args:
        obs: State vector, at least 8 dimensions. Uses only obs[:8]:
            [0] x position (normalized)
            [1] y position (normalized)
            [2] x velocity
            [3] y velocity
            [4] angle (radians)
            [5] angular velocity
            [6] left leg contact (0/1)
            [7] right leg contact (0/1)

    Returns:
        np.array([main_thrust, side_thrust]), each in [-1, 1].
        main_thrust > 0 fires main engine, side_thrust controls lateral engines.
    """
    s = obs[:8]

    # --- Angle target ---
    # Tilt toward the landing pad based on horizontal position and velocity.
    # The 0.5 and 1.0 gains create a PD controller for horizontal correction:
    #   - 0.5 * x: proportional term (steer toward center)
    #   - 1.0 * vx: derivative term (dampen horizontal velocity)
    # Clipped to ±0.4 rad to prevent excessive banking.
    angle_targ = np.clip(s[0] * 0.5 + s[2] * 1.0, -0.4, 0.4)

    # --- Hover target ---
    # Target altitude scales with horizontal distance from pad.
    # When far from center (|x| large), hover higher to give time to correct.
    # When directly above pad, target altitude is 0 (ground level).
    hover_targ = 0.55 * np.abs(s[0])

    # --- Control computation ---
    # angle_todo: PD controller on angle error
    #   - (angle_targ - s[4]) * 0.5: proportional (correct angle error)
    #   - s[5] * 1.0: derivative (dampen angular velocity)
    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0

    # hover_todo: PD controller on altitude error
    #   - (hover_targ - s[1]) * 0.5: proportional (correct altitude error)
    #   - s[3] * 0.5: derivative (dampen vertical velocity)
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    # --- Override on ground contact ---
    # Once either leg touches ground, stop trying to tilt and just
    # counteract any remaining vertical velocity. This prevents the
    # controller from tipping over during final touchdown.
    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = -s[3] * 0.5

    # --- Map to continuous actions ---
    # Scale up the control signals to action range [-1, 1].
    # The *20 gains and -1 offset on hover are tuned for default physics.
    # hover_todo * 20 - 1: the -1 bias means main engine fires only when
    # hover_todo > 0.05 (approximately), preventing unnecessary fuel burn.
    a = np.array([hover_todo * 20 - 1, -angle_todo * 20], dtype=np.float32)
    return np.clip(a, -1, 1)
