"""Shared evaluation logic for Lunar Lander RL agents.

Used by both training-time eval (OutcomeEvalCallback) and the CLI eval
script (eval_agent.py). Single source of truth for outcome classification,
per-episode metric extraction, and summary computation.

Also provides agent-loading utilities (resolve_model_path, load_training_config,
make_env_factory, build_eval_batches) shared by eval_agent.py and
collect_trajectories.py.

Key function: evaluate_agent() runs N episodes and returns structured
results with ground truth outcomes from the env's info dict.
"""

import json
import os

import numpy as np
from typing import Callable, Optional

import gymnasium

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig


# Valid outcomes from the env's info dict + timeout (from truncation).
VALID_OUTCOMES = ("landed", "crashed", "out_of_bounds", "timeout")


# ---------------------------------------------------------------------------
# Agent loading helpers — shared by eval_agent.py and collect_trajectories.py
# ---------------------------------------------------------------------------


def resolve_model_path(checkpoint_dir: str, model_name: str | None = None) -> str:
    """Resolve the path to a trained model .zip file.

    When model_name is None, looks for model.zip in the checkpoint dir.
    When model_name is a filename, checks checkpoints/ subdir first
    (new layout from train_rl.py), then top-level (old layout).

    Raises FileNotFoundError if no model is found.
    """
    if model_name:
        if os.path.isabs(model_name) or os.sep in model_name:
            # Full path provided
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Model not found: {model_name}")
            return model_name
        # Try checkpoints/ subdir first, then top-level
        subdir_path = os.path.join(checkpoint_dir, "checkpoints", model_name)
        toplevel_path = os.path.join(checkpoint_dir, model_name)
        if os.path.exists(subdir_path):
            return subdir_path
        elif os.path.exists(toplevel_path):
            return toplevel_path
        else:
            raise FileNotFoundError(
                f"Model not found: tried {subdir_path} and {toplevel_path}"
            )
    else:
        # Check: top-level model.zip, then best/model.zip (SB3 EvalCallback layout)
        model_path = os.path.join(checkpoint_dir, "model.zip")
        if os.path.exists(model_path):
            return model_path
        best_path = os.path.join(checkpoint_dir, "best", "model.zip")
        if os.path.exists(best_path):
            return best_path
        raise FileNotFoundError(f"No model.zip found in {checkpoint_dir}")


def resolve_vec_normalize_path(
    checkpoint_dir: str,
    model_path: str,
) -> str | None:
    """Resolve the matching VecNormalize stats .pkl file.

    For periodic checkpoints (rl_model_100000_steps.zip), the matching
    stats file is rl_model_vecnormalize_100000_steps.pkl.
    For the final model (model.zip), it's vec_normalize.pkl.

    Returns None if no stats file exists (agent runs without normalization).
    """
    model_basename = os.path.basename(model_path)
    if model_basename.startswith("rl_model_") and model_basename != "rl_model.zip":
        vec_norm_name = model_basename.replace(
            "rl_model_", "rl_model_vecnormalize_"
        ).replace(".zip", ".pkl")
        vec_norm_path = os.path.join(os.path.dirname(model_path), vec_norm_name)
    else:
        # Check same dir as model (e.g., best/vec_normalize.pkl), then top-level
        model_dir = os.path.dirname(model_path)
        same_dir_path = os.path.join(model_dir, "vec_normalize.pkl")
        top_level_path = os.path.join(checkpoint_dir, "vec_normalize.pkl")
        if os.path.exists(same_dir_path):
            return same_dir_path
        vec_norm_path = top_level_path

    if os.path.exists(vec_norm_path):
        return vec_norm_path
    return None


def load_training_config(checkpoint_dir: str) -> dict:
    """Load config.json from a training checkpoint directory.

    config.json is created by train_rl.py and contains variant, algo,
    history_k, n_rays, total_steps, and other training parameters.

    Raises FileNotFoundError if config.json doesn't exist.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found in {checkpoint_dir}. "
            f"This file is created by train_rl.py and is required."
        )
    with open(config_path) as f:
        return json.load(f)


def make_env_factory(
    variant: str,
    n_rays: int = 7,
    history_k: int = 8,
    profile: str | None = None,
    render_mode: str | None = None,
) -> Callable[[int], gymnasium.Env]:
    """Create an env factory function for a given variant configuration.

    Returns a callable(seed) -> wrapped env that builds the correct
    wrapper stack (DomainRandomization, PhysicsBlind, Raycast, HistoryStack)
    for the specified variant.
    """
    from lunar_lander.src.wrappers import make_lunar_lander_env

    def env_fn(seed):
        return make_lunar_lander_env(
            variant=variant,
            seed=seed,
            history_k=history_k,
            n_rays=n_rays,
            profile=profile,
            render_mode=render_mode,
        )

    return env_fn


def build_eval_batches(
    variant: str,
    n_rays: int,
    history_k: int,
    profiles_str: str | None,
    default_env_fn: Callable[[int], gymnasium.Env],
    render_mode: str | None = None,
) -> list[tuple[str, Callable[[int], gymnasium.Env]]]:
    """Build list of (profile_name, env_fn) pairs for evaluation.

    When profiles_str is None, returns a single ("default", default_env_fn) batch.
    When profiles_str is "easy,medium,hard", returns one batch per profile,
    each with its own env factory configured for that profile.
    """
    if profiles_str:
        batches = []
        for prof_name in profiles_str.split(","):
            prof_name = prof_name.strip()
            # Each profile gets its own env factory
            fn = make_env_factory(
                variant=variant,
                n_rays=n_rays,
                history_k=history_k,
                profile=prof_name,
                render_mode=render_mode,
            )
            batches.append((prof_name, fn))
        return batches
    else:
        return [("default", default_env_fn)]


def wrap_env_fn_with_corruption(
    env_fn: Callable[[int], gymnasium.Env],
    corruption_type: str,
    corruption_sigma: float = 0.1,
    training_means: np.ndarray | None = None,
) -> Callable[[int], gymnasium.Env]:
    """Wrap an env factory to add label corruption.

    Returns a new env factory that applies LabelCorruptionWrapper after
    the base env is created. The corruption seed is derived from the
    env seed for reproducibility.

    This is a factory-wrapping pattern — it takes an env_fn and returns
    a new env_fn with corruption applied. Used by collect_trajectories.py
    when --corruption is specified.

    Args:
        env_fn: Base env factory (seed -> wrapped env).
        corruption_type: "zero", "shuffle", "mean", or "noise".
        corruption_sigma: Noise std fraction for "noise" mode.
        training_means: 7-element array for "mean" mode.

    Returns:
        New env factory (seed -> wrapped env with corruption).
    """
    from lunar_lander.src.label_corruption import LabelCorruptionWrapper

    def corrupted_env_fn(seed):
        env = env_fn(seed)
        return LabelCorruptionWrapper(
            env,
            corruption_type=corruption_type,
            seed=seed,
            training_means=training_means,
            sigma=corruption_sigma,
        )

    return corrupted_env_fn


def compute_episode_metrics(
    record: dict, episode_idx: int = 0, profile: str | None = None
) -> dict:
    """Extract flat metrics dict from a single episode record.

    Reads the ground truth outcome from info["outcome"] (set by the env).
    Truncated episodes (outcome=None) are classified as "timeout".

    Args:
        record: Episode record from run_episodes() with keys:
            reward, steps, info (containing outcome + physics_config).
        episode_idx: Sequential episode index.
        profile: Optional profile name (if eval used a specific profile).

    Returns:
        Flat dict with all per-episode metrics.
    """
    info = record.get("info", {})
    physics = info.get("physics_config", {})

    # Ground truth outcome from env. None means truncated (timeout).
    outcome = info.get("outcome")
    if outcome is None:
        outcome = "timeout"

    # Compute TWR from physics params if available.
    twr = None
    if physics:
        try:
            config = LunarLanderPhysicsConfig(
                **{k: physics[k] for k in LunarLanderPhysicsConfig.PARAM_NAMES}
            )
            twr = config.twr()
        except (KeyError, TypeError):
            pass

    metrics = {
        "episode_idx": episode_idx,
        "outcome": outcome,
        "reward": record["reward"],
        "steps": record["steps"],
        "twr": twr,
        "profile": profile,
    }

    # Flatten physics params into the metrics dict.
    for pname in LunarLanderPhysicsConfig.PARAM_NAMES:
        metrics[pname] = physics.get(pname)

    return metrics


def compute_summary(episodes: list[dict]) -> dict:
    """Compute aggregate stats from a list of per-episode metric dicts.

    Args:
        episodes: List of dicts from compute_episode_metrics().

    Returns:
        Summary dict with outcome rates, reward stats, step stats.
    """
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0}

    rewards = [ep["reward"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    outcomes = [ep["outcome"] for ep in episodes]

    n_landed = sum(1 for o in outcomes if o == "landed")
    n_crashed = sum(1 for o in outcomes if o == "crashed")
    n_out_of_bounds = sum(1 for o in outcomes if o == "out_of_bounds")
    n_timeout = sum(1 for o in outcomes if o == "timeout")

    return {
        "n_episodes": n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "median_reward": float(np.median(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_steps": float(np.mean(steps)),
        "landed_pct": 100.0 * n_landed / n,
        "crashed_pct": 100.0 * n_crashed / n,
        "out_of_bounds_pct": 100.0 * n_out_of_bounds / n,
        "timeout_pct": 100.0 * n_timeout / n,
        "n_landed": n_landed,
        "n_crashed": n_crashed,
        "n_out_of_bounds": n_out_of_bounds,
        "n_timeout": n_timeout,
    }


def _make_env_thunk(env_fn, seed):
    """Zero-arg callable for SubprocVecEnv."""

    def thunk():
        return env_fn(seed)

    return thunk


def evaluate_agent(
    model: BaseAlgorithm,
    env_fn: Callable[[int], gymnasium.Env],
    n_episodes: int,
    seed: int = 0,
    vec_normalize_path: str | None = None,
    deterministic: bool = True,
    n_envs: int = 1,
    profile: str | None = None,
) -> dict:
    """Run eval episodes and return structured results.

    This is the shared eval function used by both the training callback
    (OutcomeEvalCallback) and the CLI eval script (eval_agent.py).

    Args:
        model: Trained SB3 model.
        env_fn: Factory that takes a seed and returns a wrapped env.
        n_episodes: Number of episodes to run.
        seed: Random seed for env creation.
        vec_normalize_path: Path to VecNormalize stats (.pkl).
        deterministic: Use deterministic actions (no exploration noise).
        n_envs: Number of parallel envs (1 = sequential, >1 = SubprocVecEnv).
        profile: Optional profile name to tag episodes with.

    Returns:
        dict with:
            episodes: list of per-episode metric dicts
            summary: aggregate stats dict
    """
    # Build vectorized env.
    if n_envs > 1:
        vec_env = SubprocVecEnv(
            [_make_env_thunk(env_fn, seed + i) for i in range(n_envs)]
        )
    else:
        vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])

    # Apply VecNormalize with training stats if available.
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    episodes = []

    if n_envs == 1:
        # Sequential: simple loop, one episode at a time.
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_steps = 0

        while len(episodes) < n_episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, infos = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps += 1

            if done[0]:
                record = {
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "info": {
                        k: v for k, v in infos[0].items() if k != "terminal_observation"
                    },
                }
                metrics = compute_episode_metrics(
                    record,
                    episode_idx=len(episodes),
                    profile=profile,
                )
                episodes.append(metrics)
                ep_reward = 0.0
                ep_steps = 0
    else:
        # Parallel: track per-worker state, collect as they finish.
        obs = vec_env.reset()
        ep_rewards = [0.0] * n_envs
        ep_steps = [0] * n_envs

        while len(episodes) < n_episodes:
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(actions)

            for i in range(n_envs):
                ep_rewards[i] += float(rewards[i])
                ep_steps[i] += 1

                if dones[i] and len(episodes) < n_episodes:
                    record = {
                        "reward": ep_rewards[i],
                        "steps": ep_steps[i],
                        "info": {
                            k: v
                            for k, v in infos[i].items()
                            if k != "terminal_observation"
                        },
                    }
                    metrics = compute_episode_metrics(
                        record,
                        episode_idx=len(episodes),
                        profile=profile,
                    )
                    episodes.append(metrics)
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0

    vec_env.close()

    summary = compute_summary(episodes)
    return {"episodes": episodes, "summary": summary}


def plot_eval_summary(
    episodes: list[dict], output_dir: str, per_profile_summaries: dict | None = None
):
    """Generate summary plots from eval episodes. Saves PNGs to output_dir.

    Produces up to 4 figures:
      1. outcome_counts.png — bar chart of landed/crashed/out_of_bounds/timeout counts
      2. reward_by_outcome.png — box plot of reward distributions per outcome
      3. twr_vs_outcome.png — scatter of TWR vs outcome category, with jitter
      4. profile_breakdown.png — grouped bar chart of outcome rates per profile
         (only if per_profile_summaries has multiple profiles)
    """
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    outcomes = [e["outcome"] for e in episodes]
    twrs = [e["twr"] for e in episodes if e.get("twr") is not None]

    # Color map for outcomes.
    colors = {
        "landed": "#4CAF50",
        "crashed": "#F44336",
        "out_of_bounds": "#FF9800",
        "timeout": "#FFC107",
    }

    # --- Plot 1: Outcome counts bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = {oc: outcomes.count(oc) for oc in VALID_OUTCOMES}
    # Only show outcomes that occurred.
    shown = {k: v for k, v in counts.items() if v > 0}
    bars = ax.bar(shown.keys(), shown.values(), color=[colors[k] for k in shown.keys()])
    for bar, count in zip(bars, shown.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Episodes")
    ax.set_title(f"Outcomes (n={len(episodes)})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outcome_counts.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Reward distribution by outcome ---
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_with_data = []
    groups_with_data = []
    colors_list = []
    for oc in VALID_OUTCOMES:
        grp = [e["reward"] for e in episodes if e["outcome"] == oc]
        if grp:
            labels_with_data.append(f"{oc}\n(n={len(grp)})")
            groups_with_data.append(grp)
            colors_list.append(colors[oc])
    if groups_with_data:
        bp = ax.boxplot(
            groups_with_data, tick_labels=labels_with_data, patch_artist=True
        )
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution by Outcome")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reward_by_outcome.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: TWR vs outcome ---
    if twrs:
        fig, ax = plt.subplots(figsize=(7, 4))
        # Map outcomes to y positions for a strip plot.
        y_positions = {
            "crashed": 0.0,
            "out_of_bounds": 0.33,
            "timeout": 0.66,
            "landed": 1.0,
        }
        for oc in VALID_OUTCOMES:
            vals = [
                e["twr"]
                for e in episodes
                if e.get("twr") is not None and e["outcome"] == oc
            ]
            if vals:
                y_pos = y_positions[oc]
                jitter = np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
                ax.scatter(
                    vals,
                    [y_pos + j for j in jitter],
                    alpha=0.4,
                    s=15,
                    color=colors[oc],
                    label=oc,
                )
        ax.set_xlabel("Thrust-to-Weight Ratio")
        shown_outcomes = [
            oc for oc in VALID_OUTCOMES if any(e["outcome"] == oc for e in episodes)
        ]
        ax.set_yticks([y_positions[oc] for oc in shown_outcomes])
        ax.set_yticklabels(shown_outcomes)
        ax.set_title("TWR vs Outcome")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "twr_vs_outcome.png"), dpi=150)
        plt.close(fig)

    # --- Plot 4: Per-profile breakdown (only if multiple profiles) ---
    if per_profile_summaries and len(per_profile_summaries) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        profile_names = list(per_profile_summaries.keys())
        x = np.arange(len(profile_names))
        width = 0.2
        for i, (oc, color) in enumerate(
            [
                ("landed", colors["landed"]),
                ("crashed", colors["crashed"]),
                ("out_of_bounds", colors["out_of_bounds"]),
                ("timeout", colors["timeout"]),
            ]
        ):
            pct_key = f"{oc}_pct"
            pcts = [per_profile_summaries[p].get(pct_key, 0) for p in profile_names]
            if any(v > 0 for v in pcts):
                ax.bar(
                    x + (i - 1.5) * width,
                    pcts,
                    width,
                    label=oc.replace("_", " ").title(),
                    color=color,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(profile_names)
        ax.set_ylabel("Percentage")
        ax.set_title("Outcome Rates by Profile")
        ax.legend()
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "profile_breakdown.png"), dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {output_dir}/")
