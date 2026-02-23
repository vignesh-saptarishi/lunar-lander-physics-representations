"""Lunar Lander Gymnasium wrappers.

This module contains wrappers that transform the ParameterizedLunarLander
observation space:

  - RaycastWrapper: appends terrain ray distances to observation
  - PhysicsBlindWrapper: removes physics params (dims 8-14) from obs
  - DomainRandomizationWrapper: samples new physics config each reset
  - make_lunar_lander_env(): factory wiring the correct wrapper stack per variant

See lunar-lander-testbed.md Section 4 for terrain sensing design and
Section 8 for the full wrapper architecture.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from lunar_lander.src.raycasting import compute_terrain_rays
from lunar_lander.src.physics_config import LunarLanderPhysicsConfig
from lunar_lander.src.sampling_profiles import SamplingProfile


# Default sensing range in Box2D world units. VIEWPORT_H / SCALE ~ 13.33.
DEFAULT_MAX_RANGE_WORLD = 13.33


class RaycastWrapper(gymnasium.ObservationWrapper):
    """Append terrain ray distances to the observation vector.

    Casts a fan of downward-pointing rays from the lander and intersects
    them with the terrain surface. Each ray returns a normalized distance
    in [0, 1]: 0 = ground contact, 1 = nothing within sensing range.

    Reads lander position and angle directly from the unwrapped env
    (Box2D world coordinates) and terrain segments from
    env.unwrapped.terrain_segments. This avoids denormalization errors.

    The ray parameters (n_rays, arc, range, frame) are set at wrapper
    construction and stay fixed for the env's lifetime. These are
    training-time hyperparameters — change them by wrapping differently,
    not by re-collecting data.

    Args:
        env: A ParameterizedLunarLander (or wrapped version of one).
        n_rays: Number of rays in the downward fan (default 7).
        arc_degrees: Total angular spread of the fan (default 120 deg).
        max_range: Sensing distance in world units (default 13.33 ~ viewport height).
        frame: "ego" (rays rotate with lander) or "world" (fixed downward).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        n_rays: int = 7,
        arc_degrees: float = 120.0,
        max_range: float = DEFAULT_MAX_RANGE_WORLD,
        frame: str = "ego",
    ):
        super().__init__(env)
        assert isinstance(
            env.observation_space, spaces.Box
        ), f"RaycastWrapper expects Box obs space, got {type(env.observation_space)}"
        assert frame in (
            "ego",
            "world",
        ), f"frame must be 'ego' or 'world', got '{frame}'"

        self.n_rays = n_rays
        self.arc_degrees = arc_degrees
        self.max_range = max_range
        self.frame = frame

        # Extend observation space: original dims + n_rays
        orig_low = env.observation_space.low
        orig_high = env.observation_space.high
        ray_low = np.zeros(n_rays, dtype=np.float32)
        ray_high = np.ones(n_rays, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([orig_low, ray_low]),
            high=np.concatenate([orig_high, ray_high]),
            dtype=np.float32,
        )

    def observation(self, obs):
        """Append ray distances to the base observation."""
        # Read lander state directly from Box2D (world coords).
        unwrapped = self.env.unwrapped
        lander = unwrapped.lander

        if lander is None:
            # Before first reset — return max-distance rays
            rays = np.ones(self.n_rays, dtype=np.float32)
        else:
            rays = compute_terrain_rays(
                lander_x=float(lander.position.x),
                lander_y=float(lander.position.y),
                lander_angle=float(lander.angle),
                terrain_segments=unwrapped.terrain_segments,
                n_rays=self.n_rays,
                arc_degrees=self.arc_degrees,
                max_range=self.max_range,
                frame=self.frame,
            )

        return np.concatenate([obs, rays])


class PhysicsBlindWrapper(gymnasium.ObservationWrapper):
    """Remove physics parameters (dims 8-14) from the observation.

    The full Lunar Lander observation is (15,):
      - dims 0-7: kinematic state (x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
      - dims 8-14: physics params (gravity, thrust, density, damping, wind, turbulence)

    This wrapper strips the physics params, leaving only the 8D kinematic
    state. For the "blind" agent variant, this forces the agent to infer
    physics from behavioral cues (how the lander responds to thrust) rather
    than reading the params directly.

    Applied BEFORE RaycastWrapper, so the obs flow is:
      env (15D) -> PhysicsBlind (8D) -> Raycast (8D + n_rays)
    """

    # Number of kinematic dimensions to keep (indices 0-7)
    KINEMATIC_DIMS = 8

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, spaces.Box
        ), f"PhysicsBlindWrapper expects Box obs, got {type(env.observation_space)}"
        orig_dim = env.observation_space.shape[0]
        assert (
            orig_dim >= 15
        ), f"Expected at least 15-dim obs (full Lunar Lander state), got {orig_dim}"
        self.observation_space = spaces.Box(
            low=env.observation_space.low[: self.KINEMATIC_DIMS],
            high=env.observation_space.high[: self.KINEMATIC_DIMS],
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return obs[: self.KINEMATIC_DIMS]


class DomainRandomizationWrapper(gymnasium.Wrapper):
    """Sample a new physics configuration on each episode reset.

    On each reset(), this wrapper generates a random LunarLanderPhysicsConfig
    and applies it to the underlying env before delegating to the env's reset
    (which rebuilds the Box2D world with the new physics params).

    This is the core domain randomization mechanism for RL training -- the agent
    faces a different physics regime every episode, forcing it to generalize
    across the parameter space.

    Supports two modes:
      1. Profile-based (preferred): pass a SamplingProfile that defines
         per-parameter ranges and TWR constraints. See sampling_profiles.py.
      2. Legacy twr_range: simple TWR filtering with full parameter ranges.

    Args:
        env: A ParameterizedLunarLander instance (or wrapped version).
        seed: Random seed for reproducible config sampling.
        profile: Optional SamplingProfile controlling parameter ranges.
        twr_range: Optional (min_twr, max_twr) constraint. Cannot be
            used together with profile.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        seed: int = 0,
        profile: SamplingProfile | None = None,
        twr_range: tuple[float, float] | None = None,
    ):
        super().__init__(env)
        if profile is not None and twr_range is not None:
            raise ValueError(
                "Cannot specify both 'profile' and 'twr_range'. "
                "Use profile.twr_range for TWR constraints within a profile."
            )
        self._rng = np.random.default_rng(seed)
        self._profile = profile
        self._twr_range = twr_range

    def reset(self, **kwargs):
        if self._profile is not None:
            # Profile-based sampling — delegates range overrides and TWR
            # constraint to the profile's sample() method.
            new_config = self._profile.sample(rng=self._rng)
        else:
            # Legacy path — uniform from full ranges with optional TWR filter.
            new_config = LunarLanderPhysicsConfig.randomize(
                rng=self._rng,
                twr_range=self._twr_range,
            )
        # Apply to the underlying env. ParameterizedLunarLander.reset()
        # creates a new Box2D world using self._physics_config, so setting
        # it before reset is sufficient.
        self.env.unwrapped._physics_config = new_config
        return self.env.reset(**kwargs)

    def set_profile(self, profile):
        """Swap the active sampling profile for future resets.

        Called by CurriculumCallback via SubprocVecEnv.env_method() to
        change the physics distribution mid-training. Each subprocess
        independently loads the profile from YAML when given a string name.

        Clears any legacy twr_range, since profile-based and twr_range-based
        sampling are mutually exclusive.

        Args:
            profile: Profile name (string, resolved via SamplingProfile.load())
                or a SamplingProfile object.
        """
        if isinstance(profile, str):
            self._profile = SamplingProfile.load(profile)
        else:
            self._profile = profile
        # Clear legacy twr_range — profile takes over
        self._twr_range = None


def make_lunar_lander_env(
    variant: str,
    seed: int = 0,
    history_k: int = 8,
    n_rays: int = 7,
    arc_degrees: float = 120.0,
    max_range: float = DEFAULT_MAX_RANGE_WORLD,
    ray_frame: str = "ego",
    twr_range: tuple[float, float] | None = None,
    profile: str | SamplingProfile | None = None,
    render_mode: str | None = None,
) -> gymnasium.Env:
    """Build a fully wrapped Lunar Lander env for RL training.

    Applies the correct wrapper stack for each agent variant. This is the
    factory function passed to rl_common.training.train() -- it gets called
    once per SubprocVecEnv worker.

    Wrapper order:
      1. DomainRandomizationWrapper -- new physics each reset
      2. PhysicsBlindWrapper (blind/history only) -- strips physics dims
      3. RaycastWrapper -- appends terrain ray distances
      4. HistoryStackWrapper (history only) -- stacks K frames

    Args:
        variant: "labeled", "blind", or "history".
        seed: Random seed for domain randomization and env.
        history_k: Number of obs to stack for 'history' variant.
        n_rays: Number of terrain-sensing rays.
        arc_degrees: Angular spread of the ray fan.
        max_range: Max sensing distance in Box2D world units.
        ray_frame: "ego" (rays rotate with lander) or "world" (fixed).
        twr_range: Optional (min_twr, max_twr) constraint on thrust-to-weight
            ratio for domain randomization. Rejection-samples physics configs
            until TWR falls within this range. Use (3.0, float('inf')) for
            "at least TWR 3" filtering. None = no constraint (full range).
        profile: Optional sampling profile name (string) or SamplingProfile
            object for domain randomization. When set, overrides twr_range.
            String names are resolved via SamplingProfile.load() (builtin
            name or file path). See sampling_profiles.py.

    Returns:
        A wrapped Gymnasium env ready for SB3.
    """
    from lunar_lander.src.env import ParameterizedLunarLander

    valid_variants = ("labeled", "blind", "history")
    if variant not in valid_variants:
        raise ValueError(f"variant must be one of {valid_variants}, got '{variant}'")

    # Resolve profile: string name -> SamplingProfile object
    resolved_profile = None
    if profile is not None:
        if isinstance(profile, str):
            resolved_profile = SamplingProfile.load(profile)
        else:
            resolved_profile = profile

    # Build base env with default physics (will be overwritten by DomainRandomization)
    env = ParameterizedLunarLander(render_mode=render_mode)

    # 1. Domain randomization -- new physics config every reset.
    #    Profile takes precedence over twr_range if both are given.
    if resolved_profile is not None:
        env = DomainRandomizationWrapper(env, seed=seed, profile=resolved_profile)
    else:
        env = DomainRandomizationWrapper(env, seed=seed, twr_range=twr_range)

    # 2. Physics blinding (blind + history variants)
    #    Removes dims 8-14 (physics params), keeping only kinematic state.
    #    Applied BEFORE raycasting so rays get appended to the blinded obs.
    if variant in ("blind", "history"):
        env = PhysicsBlindWrapper(env)

    # 3. Raycasting -- adds terrain ray distances to obs
    #    Reads lander pose from Box2D directly, so it works regardless of
    #    whether PhysicsBlind has been applied.
    env = RaycastWrapper(
        env,
        n_rays=n_rays,
        arc_degrees=arc_degrees,
        max_range=max_range,
        frame=ray_frame,
    )

    # 4. History stacking (history variant only)
    #    Stacks K frames so the agent can infer physics from trajectory context.
    if variant == "history":
        from rl_common.wrappers import HistoryStackWrapper

        env = HistoryStackWrapper(env, k=history_k)

    # 5. Time limit — matches Gymnasium LunarLander-v3 default (1000 steps).
    #    Without this, untrained agents can float for thousands of steps,
    #    accumulating massive fuel/shaping penalties and eventually drifting
    #    out of bounds. TimeLimit sets truncated=True (not terminated) so
    #    the value bootstrap in PPO/SAC works correctly.
    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env
