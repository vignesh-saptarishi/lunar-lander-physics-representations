"""Shared RL inference library.

Provides `run_episodes()` for running trained agents and collecting
episode-level records. Used for:
  - Evaluation (mean reward, completion rate)
  - RL data collection (expert trajectories for world model training)
  - Representation probing (collecting hidden activations alongside states)
  - Video recording of agent behavior

This module is environment-agnostic. Environment-specific metrics
(e.g., completion rate, death cause) come from the env's info dict.
"""

import os
from typing import Callable, Optional

import numpy as np
import gymnasium
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from rl_common.training import _make_env_thunk


def run_episodes(
    model: BaseAlgorithm,
    env_fn: Callable[[int], gymnasium.Env],
    n_episodes: int,
    seed: int = 0,
    vec_normalize_path: Optional[str] = None,
    deterministic: bool = True,
    collect_trajectory: bool = False,
) -> list[dict]:
    """Run a trained agent for N episodes and collect results.

    Each episode produces a record dict containing:
      - 'reward': total episode reward
      - 'steps': number of steps in the episode
      - 'info': the info dict from the terminal step (contains env-specific
        metrics like completion status, death cause, dynamics type, etc.)
      - 'trajectory' (if collect_trajectory=True): dict with keys
        'observations', 'actions', 'rewards' as numpy arrays

    The function uses a single env (not vectorized) for simplicity and
    to ensure each episode runs to natural completion without SB3's
    auto-reset interleaving.

    Args:
        model: A trained SB3 model (PPO or SAC).
        env_fn: Factory that takes a seed and returns a wrapped env.
            Should apply the same wrappers used during training.
        n_episodes: Number of episodes to run.
        seed: Random seed for the env.
        vec_normalize_path: Path to saved VecNormalize stats (.pkl).
            If provided, observations are normalized using training-time
            statistics (critical for correct inference with normalized agents).
        deterministic: If True, use deterministic actions (no exploration noise).
        collect_trajectory: If True, record full obs/action/reward sequences.

    Returns:
        List of episode record dicts.
    """
    # Build a single-env VecEnv (SB3 models require VecEnv for predict())
    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])

    # Apply VecNormalize with training-time stats if provided.
    # Without this, the agent sees unnormalized obs and produces garbage actions.
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # don't update running stats
        vec_env.norm_reward = False  # report raw rewards

    episodes = []
    obs = vec_env.reset()

    # Track current episode data
    episode_reward = 0.0
    episode_steps = 0
    traj_obs, traj_actions, traj_rewards = [], [], []

    while len(episodes) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = vec_env.step(action)

        episode_reward += float(reward[0])
        episode_steps += 1

        if collect_trajectory:
            traj_obs.append(obs[0].copy())
            traj_actions.append(action[0].copy())
            traj_rewards.append(float(reward[0]))

        # SB3 VecEnv auto-resets on done. The terminal info is stored
        # in infos[0]["terminal_info"] or infos[0] depending on version.
        # DummyVecEnv stores terminal obs in infos[0]["terminal_observation"]
        # and the terminal info dict fields are in infos[0] directly.
        if done[0]:
            record = {
                "reward": episode_reward,
                "steps": episode_steps,
                "info": {
                    k: v for k, v in infos[0].items() if k != "terminal_observation"
                },
            }
            if collect_trajectory:
                record["trajectory"] = {
                    "observations": np.array(traj_obs),
                    "actions": np.array(traj_actions),
                    "rewards": np.array(traj_rewards),
                }

            episodes.append(record)

            # Reset tracking for next episode
            episode_reward = 0.0
            episode_steps = 0
            traj_obs, traj_actions, traj_rewards = [], [], []

    vec_env.close()
    return episodes


def _get_unwrapped_env(vec_env):
    """Reach through VecEnv + Gymnasium wrappers to get the base env.

    The env stack during inference looks like:
        VecNormalize → DummyVecEnv → [wrapper stack] → base env

    DummyVecEnv stores its envs in .envs[0]. From there, .unwrapped
    traverses the Gymnasium wrapper chain to the bottom.
    """
    # VecNormalize wraps DummyVecEnv — get the inner DummyVecEnv
    inner = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    # DummyVecEnv stores gym envs in .envs list
    return inner.envs[0].unwrapped


def _grab_frame(base_env):
    """Get an RGB frame from the base env, supporting both rendering patterns.

    Platformer has a custom _render_frame() that returns the current frame
    from the obs dict (populated during _get_obs). Lunar Lander uses standard
    gymnasium render() which builds and returns an rgb_array on demand.
    """
    if hasattr(base_env, "_render_frame"):
        return _grab_frame(base_env)
    return base_env.render()


def record_episode_videos(
    model: BaseAlgorithm,
    env_fn: Callable[[int], gymnasium.Env],
    output_dir: str,
    n_episodes: int = 3,
    seed: int = 0,
    vec_normalize_path: Optional[str] = None,
    deterministic: bool = True,
    fps: int = 30,
) -> list[dict]:
    """Record episodes as mp4 videos with the trained agent.

    Runs the agent through the normal wrapper stack (so actions are
    correct), but reaches through to the unwrapped env each step to
    grab rendered RGB frames. Frames are written to mp4 via imageio.

    The wrapper stack strips RGB (StateOnlyWrapper), so the agent never
    sees the frames — we're just peeking at what the base env renders.
    We temporarily set render_mode on the unwrapped env so _get_obs()
    produces real frames instead of zeros.

    Args:
        model: A trained SB3 model (PPO or SAC).
        env_fn: Factory that takes a seed and returns a wrapped env.
        output_dir: Directory to save mp4 files.
        n_episodes: Number of episodes to record.
        seed: Random seed for the env.
        vec_normalize_path: Path to VecNormalize stats (.pkl).
        deterministic: If True, use deterministic actions.
        fps: Frames per second for the output video.

    Returns:
        List of dicts with keys: 'path' (mp4 path), 'reward', 'steps'.
    """
    import imageio.v3 as iio

    os.makedirs(output_dir, exist_ok=True)

    # Build single-env VecEnv with normalization (same as run_episodes)
    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Enable rendering on the base env so _get_obs() produces real frames.
    # The wrapper stack still strips RGB before the agent sees it, but
    # now the frames exist inside the base env's obs dict for us to grab.
    base_env = _get_unwrapped_env(vec_env)
    base_env.render_mode = "rgb_array"

    results = []
    obs = vec_env.reset()

    episode_reward = 0.0
    episode_steps = 0
    frames = [_grab_frame(base_env)]  # capture initial frame

    while len(results) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = vec_env.step(action)

        episode_reward += float(reward[0])
        episode_steps += 1

        # Grab the frame from the base env (rendered during _get_obs)
        frames.append(_grab_frame(base_env))

        if done[0]:
            # Write video for this episode
            video_path = os.path.join(output_dir, f"episode_{len(results):03d}.mp4")
            iio.imwrite(
                video_path,
                np.array(frames, dtype=np.uint8),
                fps=fps,
                codec="libx264",
            )

            results.append(
                {
                    "path": video_path,
                    "reward": episode_reward,
                    "steps": episode_steps,
                }
            )

            # Reset for next episode
            episode_reward = 0.0
            episode_steps = 0
            # DummyVecEnv auto-resets; grab the new initial frame
            frames = [_grab_frame(base_env)]

    vec_env.close()
    return results
