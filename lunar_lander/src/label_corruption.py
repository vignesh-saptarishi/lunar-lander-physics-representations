"""Label corruption wrapper for eval-time observation manipulation.

Tests whether labeled RL agents actually use their physics labels by
corrupting the 7 physics observation dimensions (indices 8-14) at eval
time. Four corruption modes:

  - zero: Set all physics dims to 0. Tests: are labels load-bearing?
  - shuffle: Randomly permute physics dims within each episode.
    Tests: do specific values matter, or just having signal?
  - mean: Replace with training-set mean. Tests: is per-episode
    variation in labels informative?
  - noise: Add Gaussian noise scaled by sigma. Tests: how precise
    must labels be?

This is an eval-time-only modification — zero training cost. The wrapper
goes in the wrapper stack after RaycastWrapper but before VecNormalize,
so the agent sees corrupted physics dims through the same normalization
it trained with.

Usage:
    env = make_lunar_lander_env(variant="labeled", ...)
    env = LabelCorruptionWrapper(env, corruption_type="zero")

    # Or with noise sweep:
    env = LabelCorruptionWrapper(env, corruption_type="noise", sigma=0.1)

Scientific context: Phase A2 of the mechanistic investigation
(mechanistic-behavioral-study.md). If performance holds under
zero-out, the agent learned behavioral mode despite having labels.
If it drops, the agent is in parametric mode — actually reading
the physics parameters.
"""

import numpy as np
import gymnasium


# Physics parameter indices in the base Lunar Lander observation.
# The full observation is 15D:
#   dims 0-7: kinematic state (x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
#   dims 8-14: physics params (gravity, main_engine_power, side_engine_power,
#              lander_density, angular_damping, wind_power, turbulence_power)
#
# After RaycastWrapper adds n_rays dims, the physics indices are unchanged
# (rays are appended at the end). After PhysicsBlindWrapper strips them,
# there are no physics dims to corrupt — so this wrapper is only meaningful
# for the "labeled" variant.
PHYSICS_START = 8
PHYSICS_END = 15  # exclusive — dims 8,9,10,11,12,13,14
N_PHYSICS_DIMS = PHYSICS_END - PHYSICS_START  # 7


class LabelCorruptionWrapper(gymnasium.ObservationWrapper):
    """Corrupt physics observation dimensions to test label dependency.

    Applied at eval time to labeled agents (22D observation = 8 kinematic
    + 7 physics + 7 rays). Modifies dims 8-14 (the physics labels) while
    leaving kinematic state (0-7) and ray distances (15+) untouched.

    Corruption is deterministic given the seed, ensuring reproducible
    experiments. For shuffle mode, the permutation is fixed per episode
    (regenerated on each reset) so all steps within an episode see the
    same shuffled labels.

    Args:
        env: A wrapped Lunar Lander env (any obs dimensionality >= 15).
        corruption_type: One of "zero", "shuffle", "mean", "noise".
        seed: Random seed for reproducible corruption (shuffle + noise).
        training_means: Array of 7 floats — per-dim training-set means,
            in PARAM_NAMES order. Required for "mean" mode.
        sigma: Noise standard deviation as a fraction of each param's
            range. Only used for "noise" mode. Default 0.1 = 10% of range.
    """

    VALID_TYPES = ("zero", "shuffle", "mean", "noise")

    def __init__(
        self,
        env: gymnasium.Env,
        corruption_type: str,
        seed: int = 0,
        training_means: np.ndarray | None = None,
        sigma: float = 0.1,
    ):
        super().__init__(env)

        if corruption_type not in self.VALID_TYPES:
            raise ValueError(
                f"corruption_type must be one of {self.VALID_TYPES}, "
                f"got '{corruption_type}'"
            )

        obs_dim = env.observation_space.shape[0]
        if obs_dim < 15:
            raise ValueError(
                f"LabelCorruptionWrapper requires obs dim >= 15 (got {obs_dim}). "
                f"This wrapper is only meaningful for labeled agents — blind "
                f"agents have physics dims already stripped."
            )

        if corruption_type == "mean" and training_means is None:
            raise ValueError(
                "corruption_type='mean' requires training_means array "
                "(7 floats, one per physics param in PARAM_NAMES order)."
            )

        self.corruption_type = corruption_type
        self._rng = np.random.default_rng(seed)
        self._training_means = training_means
        self._sigma = sigma

        # Per-episode shuffle permutation — regenerated on each reset().
        # Stays fixed across all steps within one episode so the agent
        # sees a consistent (but wrong) set of physics labels.
        self._shuffle_perm = np.arange(N_PHYSICS_DIMS)

        # Param ranges for noise scaling. Imported here to avoid circular
        # imports at module level.
        from lunar_lander.src.physics_config import LunarLanderPhysicsConfig

        self._param_ranges = np.array(
            [
                LunarLanderPhysicsConfig.RANGES[name][1]
                - LunarLanderPhysicsConfig.RANGES[name][0]
                for name in LunarLanderPhysicsConfig.PARAM_NAMES
            ],
            dtype=np.float32,
        )
        self._param_lows = np.array(
            [
                LunarLanderPhysicsConfig.RANGES[name][0]
                for name in LunarLanderPhysicsConfig.PARAM_NAMES
            ],
            dtype=np.float32,
        )
        self._param_highs = np.array(
            [
                LunarLanderPhysicsConfig.RANGES[name][1]
                for name in LunarLanderPhysicsConfig.PARAM_NAMES
            ],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        """Reset env and regenerate per-episode corruption state."""
        obs, info = self.env.reset(**kwargs)

        # Generate new shuffle permutation for this episode.
        # Other corruption types don't need per-episode state.
        if self.corruption_type == "shuffle":
            self._shuffle_perm = self._rng.permutation(N_PHYSICS_DIMS)

        return self.observation(obs), info

    def observation(self, obs):
        """Apply corruption to physics dims (indices 8-14).

        Returns a copy — never modifies the original observation array.
        Kinematic dims (0-7) and any dims beyond 14 (rays, etc.) are
        left untouched.
        """
        corrupted = obs.copy()

        if self.corruption_type == "zero":
            # Zero-out: set all physics dims to 0.
            # The simplest test: if the agent doesn't degrade, it never
            # looked at these dims. If it degrades, they're load-bearing.
            corrupted[PHYSICS_START:PHYSICS_END] = 0.0

        elif self.corruption_type == "shuffle":
            # Shuffle: permute the 7 physics values using the per-episode
            # permutation. Tests whether the agent cares about which dim
            # is which, or just that there's some signal present.
            physics = obs[PHYSICS_START:PHYSICS_END]
            corrupted[PHYSICS_START:PHYSICS_END] = physics[self._shuffle_perm]

        elif self.corruption_type == "mean":
            # Mean-replace: set to training-set mean for each dim.
            # Tests whether per-episode variation in labels is informative.
            # If performance holds, the agent only uses the mean-level signal
            # (which is constant across episodes and thus uninformative).
            corrupted[PHYSICS_START:PHYSICS_END] = self._training_means

        elif self.corruption_type == "noise":
            # Noise: add Gaussian noise scaled by sigma * param_range.
            # sigma=0.1 means noise std is 10% of each param's full range.
            # Clip to valid ranges to avoid impossible physics values.
            physics = obs[PHYSICS_START:PHYSICS_END]
            noise = self._rng.normal(
                0,
                self._sigma * self._param_ranges,
            ).astype(np.float32)
            corrupted[PHYSICS_START:PHYSICS_END] = np.clip(
                physics + noise,
                self._param_lows,
                self._param_highs,
            )

        return corrupted


def compute_training_means(trajectory_dir: str) -> np.ndarray:
    """Compute per-physics-param means from trajectory .npz files.

    Reads the physics_config from each episode's metadata and computes
    the mean value for each of the 7 physics parameters. Used to provide
    training_means for the "mean" corruption mode.

    The mean is computed across episodes (one physics config per episode),
    not across timesteps (physics is constant within an episode).

    Args:
        trajectory_dir: Directory containing episode_NNNN.npz files.

    Returns:
        np.ndarray of shape (7,) — per-param means in PARAM_NAMES order.

    Raises:
        FileNotFoundError: If no .npz files found in the directory.
    """
    import json as _json
    from pathlib import Path
    from lunar_lander.src.physics_config import LunarLanderPhysicsConfig

    npz_files = sorted(Path(trajectory_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz trajectory files found in {trajectory_dir}")

    # Collect physics param values from each episode's metadata.
    # Each episode has one physics config (constant within the episode).
    all_params = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        metadata = _json.loads(str(data["metadata_json"]))
        physics = metadata["physics_config"]
        params = [physics[name] for name in LunarLanderPhysicsConfig.PARAM_NAMES]
        all_params.append(params)

    return np.mean(all_params, axis=0).astype(np.float32)
