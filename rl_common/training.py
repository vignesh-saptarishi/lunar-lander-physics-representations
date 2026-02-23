"""Shared RL training library using Stable-Baselines3.

Provides a `train()` function that handles:
  - SubprocVecEnv construction (parallel env workers for GPU-fed batching)
  - VecNormalize for running-mean obs/reward normalization
  - PPO or SAC algorithm setup with sensible defaults
  - Periodic checkpointing and evaluation callbacks
  - Model + normalization stats saving for reproducible inference

This module is environment-agnostic. It takes an env factory function
and algorithm config, and produces a trained model. Platformer-specific
and Lunar Lander-specific wrappers are applied before calling train().

Usage:
    from rl_common.training import train, load_trained_agent

    model, vec_env = train(
        env_fn=lambda seed: make_my_env(seed),
        algo="ppo",
        total_steps=1_000_000,
    )
"""

import os
from typing import Callable, Optional

import gymnasium
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize


# -----------------------------------------------------------------------
# Default hyperparameters
# -----------------------------------------------------------------------

# PPO defaults tuned for continuous control tasks.
# Based on RL Zoo tuned hyperparams for LunarLander-v3:
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# Key change from plan 05 defaults: n_epochs 10→4, gamma 0.99→0.999.
# The old n_epochs=10 caused entropy collapse (entropy drifting from -3 toward 0+)
# because too many SGD passes on each rollout overshoot the policy.
PPO_DEFAULTS = dict(
    learning_rate=3e-4,
    n_steps=1024,  # rollout buffer per env (1024 * 8 envs = 8K steps/update)
    batch_size=64,  # mini-batch size for SGD within each epoch
    n_epochs=4,  # passes over rollout buffer per update (RL Zoo: 4, not 10)
    gamma=0.999,  # discount factor — long horizon for shaped rewards
    gae_lambda=0.98,  # GAE advantage estimation smoothing
    clip_range=0.2,  # PPO clipping — prevents large policy updates
    ent_coef=0.01,  # entropy bonus for exploration
    max_grad_norm=0.5,  # gradient clipping for stability
    vf_coef=0.5,  # value function loss weight
)

# SAC defaults for off-policy learning with replay buffer.
# gamma=0.999 matches PPO for fair comparison across algorithms.
SAC_DEFAULTS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,  # replay buffer — 1M transitions
    batch_size=256,  # larger batches stabilize off-policy learning
    tau=0.005,  # soft target update rate
    gamma=0.999,  # discount factor — match PPO for fair comparison
    train_freq=1,  # update every env step
    gradient_steps=1,  # one gradient step per env step
    ent_coef="auto",  # automatic entropy tuning — adapts exploration
    learning_starts=10_000,  # random exploration before training begins
)

# MLP policy architecture: 3 hidden layers of 256 units each.
# Deliberately simple — the scientific question is about observation content,
# not model capacity. These layers are what plan 05b probes for physics.
DEFAULT_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256, 256],  # 3 hidden layers for policy network
        vf=[256, 256, 256],  # 3 hidden layers for value function (PPO)
        # SAC uses separate Q-networks with same arch
    ),
)


# -----------------------------------------------------------------------
# Video recording callback
# -----------------------------------------------------------------------


class VideoRecorderCallback(BaseCallback):
    """Record agent episodes as mp4 at regular intervals during training.

    Uses record_episode_videos() from rl_common.inference to capture frames
    from the unwrapped env while the agent acts through the normal wrapper
    stack. Videos are saved to {video_dir}/step_{N}/ with one mp4 per episode.

    Args:
        env_fn: Factory that takes a seed and returns a wrapped env.
        video_dir: Base directory for video output.
        video_freq: Record every N timesteps.
        n_episodes: Number of episodes to record each time.
        vec_normalize_path: Path to VecNormalize stats (updated each recording).
        deterministic: Use deterministic actions for recording.
        fps: Video framerate.
    """

    def __init__(
        self,
        env_fn: Callable[[int], "gymnasium.Env"],
        video_dir: str,
        video_freq: int,
        n_episodes: int = 2,
        checkpoint_dir: str = "checkpoints/",
        deterministic: bool = True,
        fps: int = 30,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.n_episodes = n_episodes
        self.checkpoint_dir = checkpoint_dir
        self.deterministic = deterministic
        self.fps = fps

    def _on_step(self) -> bool:
        # Check if it's time to record (using total timesteps across all envs)
        if self.num_timesteps % self.video_freq != 0:
            return True

        # Save current VecNormalize stats so the recording env uses them
        vec_norm_path = os.path.join(self.checkpoint_dir, "vec_normalize.pkl")
        self.training_env.save(vec_norm_path)

        # Import here to avoid circular import (inference imports from training)
        from rl_common.inference import record_episode_videos

        step_video_dir = os.path.join(self.video_dir, f"step_{self.num_timesteps}")
        results = record_episode_videos(
            self.model,
            self.env_fn,
            output_dir=step_video_dir,
            n_episodes=self.n_episodes,
            seed=self.num_timesteps,  # different seed each recording
            vec_normalize_path=vec_norm_path,
            deterministic=self.deterministic,
            fps=self.fps,
        )

        # Log summary
        rewards = [r["reward"] for r in results]
        mean_rew = np.mean(rewards)
        if self.verbose:
            print(
                f"[Video] step={self.num_timesteps}: {len(results)} episodes, "
                f"mean_reward={mean_rew:.1f}, dir={step_video_dir}"
            )

        # Log to tensorboard if available
        if self.logger:
            self.logger.record("video/mean_reward", mean_rew)
            self.logger.record("video/n_episodes", len(results))

        return True


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------


def train(
    env_fn: Callable[[int], gymnasium.Env],
    algo: str,
    total_steps: int,
    n_envs: int = 8,
    seed: int = 42,
    log_dir: str = "logs/",
    checkpoint_dir: str = "checkpoints/",
    checkpoint_freq: int = 100_000,
    eval_freq: int = 50_000,
    eval_episodes: int = 10,
    video_freq: int = 0,
    policy_kwargs: Optional[dict] = None,
    algo_kwargs: Optional[dict] = None,
    extra_callbacks: Optional[list[BaseCallback]] = None,
) -> tuple[BaseAlgorithm, VecNormalize]:
    """Train an RL agent with SB3.

    This function handles the full training pipeline:
    1. Builds a vectorized env (SubprocVecEnv) from the factory function
    2. Wraps it with VecNormalize for running-mean normalization
    3. Creates a PPO or SAC agent with MLP policy
    4. Trains with periodic evaluation and checkpointing
    5. Saves the final model + normalization stats

    Args:
        env_fn: Factory that takes a seed (int) and returns a wrapped Gymnasium
            env. Called once per parallel worker. The env should already have
            all environment-specific wrappers applied (StateOnly, ContinuousAction,
            DomainRandomization, etc).
        algo: Algorithm name — "ppo" or "sac".
        total_steps: Total environment steps for training.
        n_envs: Number of parallel env workers in SubprocVecEnv.
        seed: Global random seed for reproducibility.
        log_dir: Directory for TensorBoard logs.
        checkpoint_dir: Directory for model checkpoints.
        checkpoint_freq: Save checkpoint every N steps.
        eval_freq: Run evaluation every N steps.
        eval_episodes: Number of episodes per evaluation.
        video_freq: Record agent videos every N steps (0 = off).
        policy_kwargs: Override default policy network kwargs.
        algo_kwargs: Override default algorithm hyperparameters.
        extra_callbacks: Additional SB3 callbacks to run during training.
            Use this to inject env-specific callbacks (e.g., annotated
            video recording) without modifying rl_common.

    Returns:
        (model, vec_env) — the trained SB3 model and its VecNormalize wrapper
        (needed for inference to apply the same normalization stats).
    """
    algo = algo.lower()
    assert algo in ("ppo", "sac"), f"Unsupported algorithm: {algo}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Build vectorized training environment ---
    # SubprocVecEnv runs each env in a separate process, which lets
    # us feed batches to the GPU without GIL contention. Each worker
    # gets a unique seed derived from the global seed.
    train_vec_env = SubprocVecEnv(
        [_make_env_thunk(env_fn, seed + i) for i in range(n_envs)]
    )

    # VecMonitor tracks episode rewards and lengths. SB3's rollout/ metrics
    # (ep_rew_mean, ep_len_mean) come from Monitor's info["episode"] dict —
    # without this, those TensorBoard curves are never written.
    train_vec_env = VecMonitor(train_vec_env)

    # VecNormalize applies running-mean normalization to observations and
    # rewards. Critical because our state vector has wildly different scales:
    # position (0-3000px), velocity (-500..500px/s), physics params (60-200),
    # and binary flags (0/1). Without normalization, the policy network would
    # need to learn to handle these scale differences internally.
    train_vec_env = VecNormalize(
        train_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,  # clip normalized obs to [-10, 10]
        clip_reward=10.0,  # clip normalized reward to [-10, 10]
    )

    # --- Build separate eval environment ---
    # Eval uses its own VecNormalize that shares running stats with training
    # but doesn't update them (training=False) and doesn't normalize rewards
    # (we want raw reward for meaningful evaluation metrics).
    eval_vec_env = SubprocVecEnv(
        [_make_env_thunk(env_fn, seed + n_envs + i) for i in range(1)]
    )
    eval_vec_env = VecMonitor(eval_vec_env)
    eval_vec_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=False,  # raw rewards for evaluation
        clip_obs=10.0,
    )
    # Eval normalization stats will be synced from training env via callback

    # --- Select algorithm and merge hyperparameters ---
    pk = policy_kwargs or DEFAULT_POLICY_KWARGS.copy()
    if algo == "ppo":
        defaults = PPO_DEFAULTS.copy()
        defaults.update(algo_kwargs or {})
        model = PPO(
            "MlpPolicy",
            train_vec_env,
            policy_kwargs=pk,
            seed=seed,
            verbose=1,
            **defaults,
        )
    else:  # sac
        defaults = SAC_DEFAULTS.copy()
        defaults.update(algo_kwargs or {})
        # SAC uses separate Q-networks, so policy_kwargs need different format.
        # SB3's SAC MlpPolicy accepts net_arch as a flat list (shared for
        # actor and critic) rather than dict(pi=..., vf=...).
        sac_pk = pk.copy()
        if "net_arch" in sac_pk and isinstance(sac_pk["net_arch"], dict):
            # Convert PPO-style dict to SAC-style flat list (uses pi arch)
            sac_pk["net_arch"] = sac_pk["net_arch"]["pi"]
        model = SAC(
            "MlpPolicy",
            train_vec_env,
            policy_kwargs=sac_pk,
            seed=seed,
            verbose=1,
            **defaults,
        )

    # --- Configure logger directly into log_dir ---
    # By default SB3 creates {tensorboard_log}/PPO_1/ subdirectories.
    # We configure the logger ourselves to write directly to log_dir,
    # so --run-dir puts everything in one flat directory.
    logger = configure_logger(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # --- Callbacks ---
    # CheckpointCallback saves the model periodically so we don't lose
    # progress if training crashes or is interrupted. Intermediate checkpoints
    # go into a checkpoints/ subdirectory to keep the top-level run dir clean
    # (config.json, final model, tensorboard, eval/ stay at top level).
    intermediate_checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),  # freq is per env step
        save_path=intermediate_checkpoint_dir,
        name_prefix="rl_model",
        save_vecnormalize=True,  # save normalization stats alongside model
    )

    # EvalCallback runs the agent on a separate env periodically and logs
    # mean reward. Also saves the "best model" based on eval reward.
    # Callers can disable this by passing eval_freq=0 (e.g., when they
    # provide their own eval callback via extra_callbacks).
    callbacks = [checkpoint_cb]
    if eval_freq > 0:
        eval_cb = EvalCallback(
            eval_vec_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best/"),
            log_path=os.path.join(log_dir, "eval/"),
            eval_freq=max(eval_freq // n_envs, 1),  # freq is per env step
            n_eval_episodes=eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_cb)
    if video_freq > 0:
        video_cb = VideoRecorderCallback(
            env_fn=env_fn,
            video_dir=os.path.join(checkpoint_dir, "videos"),
            video_freq=video_freq,
            checkpoint_dir=checkpoint_dir,
            verbose=1,
        )
        callbacks.append(video_cb)

    # --- Extra callbacks from caller ---
    # Env-specific scripts can inject custom callbacks here (e.g., lunar
    # lander's annotated video callback) without modifying rl_common.
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # --- Train ---
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
    )

    # --- Save final model + normalization stats ---
    final_model_path = os.path.join(checkpoint_dir, "model")
    model.save(final_model_path)
    train_vec_env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))

    # Clean up eval env (training env stays alive, returned to caller)
    eval_vec_env.close()

    return model, train_vec_env


def _make_env_thunk(
    env_fn: Callable[[int], gymnasium.Env], seed: int
) -> Callable[[], gymnasium.Env]:
    """Create a zero-argument callable that builds an env with a given seed.

    SubprocVecEnv needs a list of callables that take no arguments.
    We capture the seed in a closure so each worker gets a unique seed.
    """

    def thunk():
        return env_fn(seed)

    return thunk


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------


def load_trained_agent(
    checkpoint_dir: str,
    algo: str = "ppo",
    env_fn: Optional[Callable[[int], gymnasium.Env]] = None,
    device: str = "auto",
) -> tuple[BaseAlgorithm, Optional[VecNormalize]]:
    """Load a trained agent from a checkpoint directory.

    The checkpoint directory should contain:
      - model.zip: the SB3 model weights and policy
      - vec_normalize.pkl: VecNormalize running statistics

    Args:
        checkpoint_dir: Path to the checkpoint directory.
        algo: Algorithm used to train ("ppo" or "sac").
        env_fn: Optional env factory for creating a VecNormalize wrapper.
            If None, returns model without VecNormalize (useful for just
            inspecting the model).
        device: Device for the model ("auto", "cpu", "cuda").

    Returns:
        (model, vec_env) — the loaded model and optionally the VecNormalize
        wrapper with training-time statistics.
    """
    algo = algo.lower()
    model_path = os.path.join(checkpoint_dir, "model.zip")
    vec_norm_path = os.path.join(checkpoint_dir, "vec_normalize.pkl")

    AlgoClass = PPO if algo == "ppo" else SAC

    # Load VecNormalize stats if available and env_fn provided.
    # We must build the env BEFORE loading the model when n_envs differs
    # from training, because SB3's load(path, env=...) handles this case
    # while set_env() requires matching n_envs.
    vec_env = None
    if env_fn is not None and os.path.exists(vec_norm_path):
        # Build a single-env VecEnv and load saved normalization stats
        vec_env = SubprocVecEnv([_make_env_thunk(env_fn, seed=0)])
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        # Set to eval mode: don't update running stats, don't normalize rewards
        vec_env.training = False
        vec_env.norm_reward = False
        # Load model with env attached — handles n_envs mismatch
        model = AlgoClass.load(model_path, env=vec_env, device=device)
    else:
        model = AlgoClass.load(model_path, device=device)

    return model, vec_env
