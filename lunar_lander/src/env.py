"""Parameterized Lunar Lander environment — fork of Gymnasium LunarLander-v3.

This module contains the core environment class. It's a modified copy of
gymnasium/envs/box2d/lunar_lander.py (~770 lines) with these changes:

  1. Physics parameterization: 7 hardcoded constants replaced by
     LunarLanderPhysicsConfig fields (gravity, thrust, density, damping, wind).

  2. Continuous-only: Discrete action space removed. Actions are always
     Box(2,) = (main_thrust, side_thrust) in [-1, 1].

  3. Wind always enabled: The enable_wind bool is removed. wind_power=0
     means no wind — simpler parameterization.

  4. Extended observation: (8,) → (15,) by appending 7 raw physics params.
     Training code decides what the agent sees via wrappers.

  5. Metadata in info: Every step/reset returns physics_config in the info dict.

The original Gymnasium source is credited to Oleg Klimov and Andrea PIERRÉ.

Coordinate system notes (important for calibration):
  - Box2D world coords: positions in meters-ish (VIEWPORT / SCALE range).
    x ∈ [0, 20], y ∈ [0, 13.3] approximately.
  - State vector coords: normalized. x,y divided by (VIEWPORT/SCALE/2),
    velocities scaled by (VIEWPORT/SCALE/2)/FPS, angular velocity scaled by 20/FPS.
  - Calibration works in world coords (body.position, body.linearVelocity)
    to avoid the normalization confusion.
"""

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run: "
        'pip install swig && pip install "gymnasium[box2d]"'
    ) from e

if TYPE_CHECKING:
    import pygame

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig


# --- Constants that are NOT parameterized ---
# These define the simulation's coordinate system, timing, and geometry.
# Changing these would break the coordinate normalization in the state vector,
# so they stay fixed. Physics behavior is varied through PhysicsConfig instead.

FPS = 50  # Physics timestep = 1/50 = 0.02 seconds
SCALE = 30.0  # Pixel-to-world coordinate scaling factor

# Initial random force applied to lander at spawn — gives each episode
# a different starting trajectory. Not parameterized because it's about
# episode variety, not physics properties.
INITIAL_RANDOM = 1000.0

# Lander body geometry in pixel coordinates (divided by SCALE for Box2D).
# This polygon defines the lander shape and, combined with density,
# determines the lander's mass and moment of inertia.
LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]

# Leg geometry and attachment constants (in pixel coords, divided by SCALE).
LEG_AWAY = 20  # Horizontal offset of leg attachment from lander center
LEG_DOWN = 18  # Vertical offset of leg attachment
LEG_W, LEG_H = 2, 8  # Leg box dimensions
LEG_SPRING_TORQUE = 40  # Joint motor torque for leg springs

# Engine position constants (in pixel coords).
SIDE_ENGINE_HEIGHT = 14  # Y offset of side engines on lander body
SIDE_ENGINE_AWAY = 12  # X offset of side engines from center
MAIN_ENGINE_Y_LOCATION = 4  # Y offset of main engine nozzle

# Viewport dimensions (pixels). Used for rendering and coordinate normalization.
VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    """Box2D contact listener that detects lander body and leg ground contacts.

    When the main lander body touches the ground, it's game over (crash).
    Leg contacts are tracked separately — both legs touching with low velocity
    means a successful landing.
    """

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class ParameterizedLunarLander(gym.Env, EzPickle):
    """Lunar Lander with configurable physics parameters.

    Fork of Gymnasium's LunarLander-v3 with 7 continuous physics parameters
    that can be varied per episode. Continuous action space only.

    Key differences from Gymnasium LunarLander-v3:
      - Constructor takes LunarLanderPhysicsConfig instead of individual params
      - Observation is (15,) = 8 base state + 7 physics params
      - Always continuous actions: Box(2,) = (main_thrust, side_thrust)
      - Wind always enabled (wind_power=0 means no wind)
      - Info dict contains physics_config on every step

    Args:
        render_mode: "human" for pygame window, "rgb_array" for pixel arrays,
            None for no rendering (fastest, use for training/collection).
        physics_config: Physics parameters. None = default config (reproduces
            standard LunarLander-v3 behavior exactly).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        physics_config: LunarLanderPhysicsConfig | None = None,
    ):
        # EzPickle stores constructor args for pickling (needed by some
        # Gymnasium wrappers and vectorized envs).
        EzPickle.__init__(self, render_mode, physics_config)

        # Store physics config — default if None provided.
        # The config is immutable for the lifetime of this env instance.
        # To change physics, create a new env (or pass config via reset options
        # in a future extension).
        self._physics_config = physics_config or LunarLanderPhysicsConfig()

        # --- Rendering state ---
        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True

        # --- Box2D world ---
        # Gravity is set from config. The world is recreated on each reset()
        # to work around a Gymnasium bug where leftover bodies cause issues
        # (see https://github.com/Farama-Foundation/Gymnasium/issues/728).
        self.world = Box2D.b2World(gravity=(0, self._physics_config.gravity))
        self.moon = None
        self.lander: Box2D.b2Body | None = None
        self.particles = []
        self.terrain_segments: list[tuple[float, float, float, float]] = []

        self.prev_reward = None

        # --- Observation space: 15 dimensions ---
        # First 8: standard LunarLander state (normalized coordinates).
        # Last 7: raw physics parameters (unnormalized — normalize in wrappers
        # at training time, not at collection time).
        #
        # Base state dims and their ranges (from Gymnasium v3):
        #   [0] x position:        [-2.5, 2.5]   (normalized by VIEWPORT_W/SCALE/2)
        #   [1] y position:        [-2.5, 2.5]   (normalized by VIEWPORT_H/SCALE/2)
        #   [2] x velocity:        [-10, 10]     (scaled by (VIEWPORT_W/SCALE/2)/FPS)
        #   [3] y velocity:        [-10, 10]     (scaled by (VIEWPORT_H/SCALE/2)/FPS)
        #   [4] angle:             [-2π, 2π]     (radians)
        #   [5] angular velocity:  [-10, 10]     (scaled by 20/FPS)
        #   [6] left leg contact:  [0, 1]        (boolean as float)
        #   [7] right leg contact: [0, 1]        (boolean as float)
        #
        # Physics param dims [8:15] use raw values with bounds from RANGES:
        #   [8]  gravity:            [-12.0, -2.0]
        #   [9]  main_engine_power:  [5.0, 25.0]
        #   [10] side_engine_power:  [0.2, 1.5]
        #   [11] lander_density:     [2.5, 10.0]
        #   [12] angular_damping:    [0.0, 5.0]
        #   [13] wind_power:         [0.0, 30.0]
        #   [14] turbulence_power:   [0.0, 5.0]
        base_low = np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, 0.0, 0.0],
            dtype=np.float32,
        )
        base_high = np.array(
            [2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0],
            dtype=np.float32,
        )

        # Build physics param bounds from the config's RANGES dict,
        # in canonical PARAM_NAMES order.
        physics_low = np.array(
            [
                LunarLanderPhysicsConfig.RANGES[n][0]
                for n in LunarLanderPhysicsConfig.PARAM_NAMES
            ],
            dtype=np.float32,
        )
        physics_high = np.array(
            [
                LunarLanderPhysicsConfig.RANGES[n][1]
                for n in LunarLanderPhysicsConfig.PARAM_NAMES
            ],
            dtype=np.float32,
        )

        low = np.concatenate([base_low, physics_low])
        high = np.concatenate([base_high, physics_high])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # --- Action space: continuous only ---
        # (main_thrust, side_thrust) both in [-1, 1].
        # Main engine: <0 = off, 0..1 = 50%..100% throttle.
        # Side engines: |action| < 0.5 = off, |action| >= 0.5 = fire left/right
        #               with throttle scaling from 50% to 100%.
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

        self.render_mode = render_mode

    @property
    def physics_config(self) -> LunarLanderPhysicsConfig:
        """Read-only access to the current physics configuration."""
        return self._physics_config

    def _destroy(self):
        """Clean up all Box2D bodies before reset or close."""
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset the environment to initial state.

        Creates fresh Box2D world, terrain, lander, and legs. Applies a random
        initial impulse for trajectory variety. Runs one no-op step to populate
        the initial observation.

        Args:
            seed: RNG seed for reproducibility.
            options: Unused currently. Reserved for future per-episode config.

        Returns:
            (observation, info) tuple. Observation is (15,) float32 array.
            Info contains 'physics_config' dict.
        """
        super().reset(seed=seed)
        self._destroy()

        # Recreate the world from scratch on every reset. This is a workaround
        # for a Gymnasium bug where leftover Box2D state causes non-determinism
        # (https://github.com/Farama-Foundation/Gymnasium/issues/728).
        self.world = Box2D.b2World(gravity=(0, self._physics_config.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        # --- Terrain generation ---
        # The terrain is a series of line segments with a flat landing pad
        # in the center. Helipad position is deterministic (always centered).
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        # Store terrain segments as (x1, y1, x2, y2) tuples for raycasting
        # and metadata. These are in Box2D world coordinates.
        self.terrain_segments = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.terrain_segments.append((p1[0], p1[1], p2[0], p2[1]))

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # --- Create lander body ---
        # The lander spawns at the top center of the viewport.
        # Density comes from physics config (default 5.0).
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                # Density from config — higher density = heavier lander.
                # Mass = density * polygon_area (computed by Box2D).
                density=self._physics_config.lander_density,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),
        )
        # Angular damping from config — 0 = free rotation (Gymnasium default),
        # higher values resist angular velocity changes (more stable).
        self.lander.angularDamping = self._physics_config.angular_damping

        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply random initial impulse — gives trajectory variety.
        # This force is NOT parameterized; it's about episode diversity,
        # not physics properties.
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # --- Wind initialization ---
        # Wind pattern indices are randomized per episode for statistical
        # independence (Gymnasium v3 fix). Wind is always "enabled" —
        # wind_power=0 in the config means the force magnitude is zero.
        self.wind_idx = self.np_random.integers(-9999, 9999)
        self.torque_idx = self.np_random.integers(-9999, 9999)

        # --- Create legs ---
        # Legs are separate Box2D bodies attached via revolute joints.
        # Leg density is fixed at 1.0 (not parameterized).
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,  # Fixed leg density — not parameterized.
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        # Run one no-op step to populate the initial state vector.
        # This is how Gymnasium LunarLander works — the first observation
        # comes from a zero-action step after body creation.
        obs = self.step(np.array([0.0, 0.0], dtype=np.float32))[0]
        # _last_obs is set by step() above — no need to set again here.

        info = {
            "physics_config": self._physics_config.to_dict(),
            "terrain_segments": [list(seg) for seg in self.terrain_segments],
        }
        return obs, info

    def _create_particle(self, mass, x, y, ttl):
        """Create a visual-only particle for engine exhaust effects.

        Particles are purely cosmetic — they don't affect physics.
        Only created when render_mode is set (see step()).
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        """Remove expired or all particles from the world."""
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        """Advance the simulation by one timestep.

        Applies wind forces, engine impulses, steps Box2D, computes reward,
        and builds the 15D observation vector.

        Args:
            action: np.array([main_thrust, side_thrust]), each in [-1, 1].
                Main engine: <0 = off, 0..1 = 50%..100% throttle.
                Side engines: |val| < 0.5 = off, direction = sign(val),
                              throttle scales 50%..100% for |val| in 0.5..1.0.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
            observation: (15,) float32 — 8 base state + 7 physics params.
            reward: float — shaped reward (see reward section below).
            terminated: bool — True if crashed, landed, or out of bounds.
            truncated: bool — always False (caller handles time limits).
            info: dict with 'physics_config'.
        """
        assert self.lander is not None, "You forgot to call reset()"

        # --- Wind forces ---
        # Applied only when airborne (neither leg touching ground).
        # Wind is a quasi-periodic function of time step index:
        #   wind_mag = tanh(sin(2k*t) + sin(π*k*t)) * wind_power
        # where k=0.01. This function is provably non-periodic, creating
        # naturalistic gusting patterns. wind_power=0 → no force.
        if not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # Horizontal wind force
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self._physics_config.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            # Angular turbulence torque — independent of horizontal wind.
            # Disturbs lander orientation, making stabilization harder.
            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self._physics_config.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        # --- Engine impulses ---
        # Continuous actions only. Clip to [-1, 1] for safety.
        action = np.clip(action, -1, +1).astype(np.float64)

        # tip/side vectors define the lander's local coordinate frame.
        # tip = (sin(angle), cos(angle)) — points "up" from lander's perspective.
        # side = (-cos(angle), sin(angle)) — points "right" from lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])

        # Small random dispersion in engine nozzle direction — adds realistic
        # noise to thrust application point.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # --- Main engine ---
        # Fires when action[0] > 0. Throttle maps [0, 1] → [0.5, 1.0].
        # The 50% minimum prevents unrealistically precise low-thrust maneuvers.
        m_power = 0.0
        if action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            assert 0.5 <= m_power <= 1.0

            # Compute impulse application point (slightly below lander center).
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (
                self.lander.position[0] + ox,
                self.lander.position[1] + oy,
            )

            if self.render_mode is not None:
                # Exhaust particles — visual only, no physics effect.
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (
                        ox * self._physics_config.main_engine_power * m_power,
                        oy * self._physics_config.main_engine_power * m_power,
                    ),
                    impulse_pos,
                    True,
                )

            # Apply thrust impulse to lander body. Negative because reaction
            # force: exhaust goes down, lander goes up.
            self.lander.ApplyLinearImpulse(
                (
                    -ox * self._physics_config.main_engine_power * m_power,
                    -oy * self._physics_config.main_engine_power * m_power,
                ),
                impulse_pos,
                True,
            )

        # --- Side engines ---
        # Fire when |action[1]| > 0.5. Direction = sign(action[1]).
        # Throttle maps [0.5, 1.0] → [0.5, 1.0].
        s_power = 0.0
        if np.abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            assert 0.5 <= s_power <= 1.0

            # Side engine impulse application point.
            # Note: There's a known bug in Gymnasium where the constant 17
            # doesn't match SIDE_ENGINE_HEIGHT=14, causing orientation-dependent
            # torque. We preserve this behavior for compatibility.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )

            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * self._physics_config.side_engine_power * s_power,
                        oy * self._physics_config.side_engine_power * s_power,
                    ),
                    impulse_pos,
                    True,
                )

            self.lander.ApplyLinearImpulse(
                (
                    -ox * self._physics_config.side_engine_power * s_power,
                    -oy * self._physics_config.side_engine_power * s_power,
                ),
                impulse_pos,
                True,
            )

        # --- Physics step ---
        # Advance Box2D by 1/FPS seconds with 6*30 velocity iterations
        # and 2*30 position iterations (Gymnasium defaults).
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # --- Build observation ---
        pos = self.lander.position
        vel = self.lander.linearVelocity

        # Base 8D state (normalized coordinates — see module docstring).
        state = [
            # Position: normalized by half-viewport in world coords.
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            # Velocity: scaled by viewport/FPS factor.
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            # Angle and angular velocity.
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            # Leg contact booleans.
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # Append 7 raw physics parameters in canonical order.
        # These are NOT normalized — training wrappers handle normalization.
        physics_params = self._physics_config.as_array()
        full_state = np.array(state, dtype=np.float32)
        full_obs = np.concatenate([full_state, physics_params])
        assert len(full_obs) == 15

        # --- Reward computation ---
        # Shaped reward from Gymnasium: penalizes distance, velocity, tilt;
        # rewards leg contact. Additional penalties for fuel use and
        # bonuses/penalties for landing/crashing.
        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Fuel penalties — encourages efficient engine use.
        reward -= m_power * 0.30  # Main engine fuel cost
        reward -= s_power * 0.03  # Side engine fuel cost (10x cheaper)

        # --- Termination ---
        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            # Crashed (body contact with ground) or drifted out of bounds.
            terminated = True
            reward = -100
        if not self.lander.awake:
            # Lander came to rest (successfully landed).
            terminated = True
            reward = +100

        # Determine outcome for terminal steps.
        # Three terminal outcomes:
        #   "landed"      — lander came to rest anywhere (awake=False)
        #   "crashed"     — body contact with ground (game_over flag)
        #   "out_of_bounds" — drifted beyond viewport (|x| >= 1.0)
        # Non-terminal steps: None.
        # Timeout (truncation by TimeLimit wrapper) is handled by eval code
        # since the env doesn't enforce time limits internally.
        outcome = None
        if terminated:
            if not self.lander.awake:
                outcome = "landed"
            elif abs(state[0]) >= 1.0:
                outcome = "out_of_bounds"
            else:
                outcome = "crashed"

        if self.render_mode == "human":
            self.render()

        info = {
            "outcome": outcome,
            "physics_config": self._physics_config.to_dict(),
            "terrain_segments": [list(seg) for seg in self.terrain_segments],
        }

        # Store the raw 15D obs for external access (e.g., eval video recording
        # needs the full state vector even when wrappers transform the obs).
        self._last_obs = full_obs

        # truncation=False — we don't enforce time limits internally.
        # Caller wraps with TimeLimit if desired.
        return full_obs, reward, terminated, False, info

    def render(self):
        """Render the current frame.

        In "human" mode, displays in a pygame window.
        In "rgb_array" mode, returns a (H, W, 3) uint8 numpy array.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run: " 'pip install "gymnasium[box2d]"'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        """Clean up pygame resources."""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
