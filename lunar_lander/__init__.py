"""Parameterized Lunar Lander environment for physics priors research.

Fork of Gymnasium's LunarLander-v3 with 7 continuous physics parameters
that can be varied per episode. Continuous action space only.

Registration makes the env available via:
    gym.make("ParameterizedLunarLander-v0", physics_config=config)
"""

from gymnasium.envs.registration import register

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig

register(
    id="ParameterizedLunarLander-v0",
    entry_point="lunar_lander.src.env:ParameterizedLunarLander",
    # No max_episode_steps here — caller wraps with TimeLimit if desired.
    # Data collection scripts set their own step limits.
)
