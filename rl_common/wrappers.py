"""Generic Gymnasium wrappers for RL training.

These wrappers are environment-agnostic — they work with any Gymnasium env
that has a Box observation space. Environment-specific wrappers (e.g., for
the platformer's Dict obs or hybrid action space) live in their respective
directories.
"""

from collections import deque

import numpy as np
import gymnasium
from gymnasium import spaces


class HistoryStackWrapper(gymnasium.ObservationWrapper):
    """Stack the last K observations into a single flat vector.

    This gives the agent a temporal context window without requiring a
    recurrent architecture. Useful for inferring dynamics from recent
    trajectory — e.g., the generalist-history agent variant uses this
    to detect which physics regime is active from recent state transitions.

    The wrapper maintains a deque of the last K observations. On reset(),
    the deque is filled with K copies of the initial observation (zero-padding
    would introduce a distributional shift that could confuse the policy
    during the first K steps). On step(), the new observation is appended
    and the concatenated stack is returned.

    Args:
        env: A Gymnasium env with a flat Box observation space.
        k: Number of observations to stack. Higher K = more temporal
           context but larger input dimension (K * obs_dim).
    """

    def __init__(self, env: gymnasium.Env, k: int = 8):
        super().__init__(env)
        assert isinstance(
            env.observation_space, spaces.Box
        ), f"HistoryStackWrapper requires Box obs space, got {type(env.observation_space)}"
        assert (
            len(env.observation_space.shape) == 1
        ), f"HistoryStackWrapper requires 1D obs, got shape {env.observation_space.shape}"

        self.k = k
        self._obs_dim = env.observation_space.shape[0]

        # Build the stacked observation space by tiling the original bounds K times.
        # This preserves the correct low/high ranges for each dimension in the stack.
        low = np.tile(env.observation_space.low, k)
        high = np.tile(env.observation_space.high, k)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

        # Internal buffer — filled with initial obs on reset
        self._history: deque = deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the entire history with the initial observation so the agent
        # sees a consistent (if repetitive) input from step 0, rather than
        # zeros that don't correspond to any real state.
        self._history.clear()
        for _ in range(self.k):
            self._history.append(obs)
        return self._stack(), info

    def observation(self, obs):
        """Called by step() via ObservationWrapper machinery."""
        self._history.append(obs)
        return self._stack()

    def _stack(self) -> np.ndarray:
        """Concatenate the history deque into a single flat vector.

        Order: [oldest_obs, ..., newest_obs]. The most recent observation
        occupies the last obs_dim elements of the output.
        """
        return np.concatenate(list(self._history), axis=0)
