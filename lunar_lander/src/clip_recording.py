"""Clip recording utilities for article video production.

Three-stage pipeline:
  1. select_episodes() — find prototypical episodes from metrics.csv
  2. record_clip() — re-record an episode with rgb_frames using fixed physics
  3. render_clip() — render clean game-view mp4 + companion JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig


def select_episodes(
    df: pd.DataFrame,
    n: int,
    outcome: str,
    sort_by: str,
    ascending: bool = False,
    filters: dict[str, tuple[float | None, float | None]] | None = None,
    diversity_on: str | None = None,
) -> list[dict]:
    """Select prototypical episodes from a metrics DataFrame.

    Filters by outcome and optional metric thresholds, then returns
    the top N episodes sorted by the specified metric. Optionally
    ensures diversity on a physics parameter (e.g. gravity) by
    binning the filtered results and picking from each bin.

    Args:
        df: DataFrame with per-episode metrics (from metrics.csv).
        n: Number of episodes to select.
        outcome: Required outcome ("landed", "crashed", "timeout").
        sort_by: Column name to sort by for prototypicality.
        ascending: Sort order. False = highest first (e.g. autocorrelation).
        filters: Optional dict of {column: (min_val, max_val)}. None means
            no bound on that side.
        diversity_on: Optional column name to diversify on. When set,
            bins the filtered results into N bins on this column and
            picks the best (by sort_by) from each bin.

    Returns:
        List of dicts, one per selected episode, with all metrics.csv
        columns as keys. Empty list if no episodes match.
    """
    filtered = df[df["outcome"] == outcome].copy()

    if filters:
        for col, (lo, hi) in filters.items():
            if lo is not None:
                filtered = filtered[filtered[col] >= lo]
            if hi is not None:
                filtered = filtered[filtered[col] <= hi]

    if filtered.empty:
        return []

    if diversity_on and diversity_on in filtered.columns and len(filtered) >= n:
        # Bin into N groups on the diversity column, pick best from each.
        filtered = filtered.sort_values(sort_by, ascending=ascending)
        filtered["_bin"] = pd.qcut(
            filtered[diversity_on],
            q=min(n, len(filtered)),
            labels=False,
            duplicates="drop",
        )
        # Take the best (by sort_by) from each bin.
        selected = filtered.groupby("_bin").first().reset_index(drop=True)
        selected = selected.head(n)
        if "_bin" in selected.columns:
            selected = selected.drop(columns=["_bin"])
    else:
        selected = filtered.sort_values(sort_by, ascending=ascending).head(n)

    return selected.to_dict("records")


def extract_physics_config(npz_path: str | Path) -> LunarLanderPhysicsConfig:
    """Extract the physics config from a trajectory .npz file.

    Reads the metadata_json field and reconstructs the
    LunarLanderPhysicsConfig that was active during that episode.

    Args:
        npz_path: Path to a trajectory .npz file.

    Returns:
        LunarLanderPhysicsConfig with the episode's physics parameters.
    """
    data = np.load(str(npz_path), allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return LunarLanderPhysicsConfig.from_dict(metadata["physics_config"])


def record_clip(
    checkpoint_dir: str | Path | None,
    physics_config: LunarLanderPhysicsConfig,
    output_path: str | Path,
    variant: str = "labeled",
    use_heuristic: bool = False,
    seed: int = 0,
    max_steps: int = 500,
    n_rays: int = 7,
    history_k: int = 8,
    corruption: str | None = None,
    corruption_sigma: float = 0.1,
) -> dict:
    """Record a single episode with rgb_frames using a fixed physics config.

    Builds the full env + wrapper stack, injects the physics config before
    reset, runs the agent (or heuristic), and saves the episode as .npz
    with rgb_frames included.

    Args:
        checkpoint_dir: Path to agent directory with model.zip, config.json,
            vec_normalize.pkl. None if use_heuristic=True.
        physics_config: The physics parameters to use for this episode.
        output_path: Where to save the .npz file.
        variant: "labeled", "blind", or "history".
        use_heuristic: If True, use a simple heuristic policy instead of
            loading a trained model. Useful for testing.
        seed: Env seed for terrain generation.
        max_steps: Maximum episode steps before timeout.
        n_rays: Number of terrain-sensing rays.
        history_k: History stack depth (for history variant).
        corruption: Optional corruption type ("zero", "shuffle", "mean", "noise").
        corruption_sigma: Noise sigma for corruption="noise".

    Returns:
        Dict with: output_path, outcome, n_steps, total_reward,
        physics_config (dict).
    """
    from lunar_lander.src.env import ParameterizedLunarLander
    from lunar_lander.src.wrappers import PhysicsBlindWrapper, RaycastWrapper
    from lunar_lander.src.episode_io import save_episode, run_episode

    # Build env with render_mode for frame capture.
    # We create the base env directly with the target physics config,
    # then apply wrappers manually (skipping DomainRandomization).
    base_env = ParameterizedLunarLander(
        render_mode="rgb_array",
        physics_config=physics_config,
    )

    # Apply the wrapper stack WITHOUT DomainRandomization.
    # Training order: DomainRandom → PhysicsBlind → Raycast → HistoryStack
    # Here we skip DomainRandom (fixed physics) but keep the rest in order.
    env = base_env
    if variant in ("blind", "history"):
        env = PhysicsBlindWrapper(env)
    env = RaycastWrapper(env, n_rays=n_rays)
    if variant == "history":
        from rl_common.wrappers import HistoryStackWrapper

        env = HistoryStackWrapper(env, k=history_k)
    # labeled: just Raycast (physics dims stay in obs)

    # Optional corruption wrapper for Grid 2 (parametric trap) clips.
    if corruption:
        from lunar_lander.src.label_corruption import LabelCorruptionWrapper

        kwargs = {}
        if corruption == "noise":
            kwargs["sigma"] = corruption_sigma
        elif corruption == "mean":
            # For mean corruption, default to zeros if no training means provided.
            kwargs["training_means"] = np.zeros(7, dtype=np.float32)
        env = LabelCorruptionWrapper(env, corruption_type=corruption, **kwargs)

    # Build policy function.
    if use_heuristic:
        # Simple constant-thrust policy for testing.
        def policy_fn(obs):
            return np.array([0.5, 0.0], dtype=np.float32)

    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from lunar_lander.src.eval_utils import (
            resolve_model_path,
            resolve_vec_normalize_path,
        )

        checkpoint_dir = Path(checkpoint_dir)
        model_path = resolve_model_path(str(checkpoint_dir))
        model = PPO.load(model_path, device="cpu")

        # Load VecNormalize stats for correct observation normalization.
        vec_norm_path = resolve_vec_normalize_path(
            str(checkpoint_dir),
            model_path,
        )

        if vec_norm_path:
            # We need to run through VecNormalize for correct obs.
            # Wrap in DummyVecEnv + VecNormalize, run manually.
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(vec_norm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

            result = _run_episode_vec(
                vec_env,
                model,
                base_env,
                seed,
                max_steps,
                physics_config,
            )
            vec_env.close()

            # Save .npz
            output_path = Path(output_path)
            save_episode(
                path=output_path,
                states=result["states"],
                actions=result["actions"],
                rewards=result["rewards"],
                dones=result["dones"],
                metadata=result["metadata"],
                rgb_frames=result["rgb_frames"],
            )
            return {
                "output_path": output_path,
                "outcome": result["metadata"]["outcome"],
                "n_steps": result["metadata"]["n_steps"],
                "total_reward": result["metadata"]["total_reward"],
                "physics_config": result["metadata"]["physics_config"],
            }

        else:

            def policy_fn(obs):
                action, _ = model.predict(
                    obs.reshape(1, -1),
                    deterministic=True,
                )
                return action[0]

    # Run episode with frame capture (heuristic or model without VecNormalize).
    result = run_episode(
        env=env,
        policy_fn=policy_fn,
        seed=seed,
        max_steps=max_steps,
        save_frames=True,
    )

    output_path = Path(output_path)
    save_episode(
        path=output_path,
        states=result["states"],
        actions=result["actions"],
        rewards=result["rewards"],
        dones=result["dones"],
        metadata=result["metadata"],
        rgb_frames=result["rgb_frames"],
    )

    env.close()

    return {
        "output_path": output_path,
        "outcome": result["metadata"]["outcome"],
        "n_steps": result["metadata"]["n_steps"],
        "total_reward": result["metadata"]["total_reward"],
        "physics_config": result["metadata"]["physics_config"],
    }


def _run_episode_vec(
    vec_env,
    model,
    base_env,
    seed: int,
    max_steps: int,
    physics_config: LunarLanderPhysicsConfig,
) -> dict:
    """Run one episode through VecNormalize stack, capturing raw state + frames.

    Same approach as eval_agent.py's _record_annotated_videos: run through
    the VecEnv for correct obs normalization, but reach through to the
    unwrapped base env for raw states and rgb_frames.
    """
    # Inject physics config and seed before reset.
    # The seed controls terrain generation — must match the original episode
    # for reproducible trajectories with deterministic policy.
    base_env._physics_config = physics_config
    vec_env.seed(seed)
    obs = vec_env.reset()

    states = [base_env._last_obs.copy()]
    actions = []
    rewards = []
    dones = []
    frames = [base_env.render()]
    total_reward = 0.0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)

        total_reward += float(reward[0])
        states.append(base_env._last_obs.copy())
        actions.append(action[0].copy())
        rewards.append(float(reward[0]))
        dones.append(bool(done[0]))
        frames.append(base_env.render())

        if done[0]:
            break

    # Classify outcome.
    final_reward = rewards[-1] if rewards else 0
    if final_reward >= 100:
        outcome = "landed"
    elif final_reward <= -100:
        outcome = "crashed"
    else:
        outcome = "timeout"

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "rgb_frames": np.array(frames, dtype=np.uint8),
        "metadata": {
            "physics_config": physics_config.to_dict(),
            "outcome": outcome,
            "seed": seed,
            "n_steps": len(actions),
            "total_reward": total_reward,
            "policy": "rl_agent",
        },
    }


def render_clean_clip(
    npz_path: str | Path,
    output_path: str | Path,
    fps: int = 50,
    variant: str = "",
    condition: str = "",
) -> tuple[Path, Path]:
    """Render a clean game-view-only mp4 clip + companion JSON metadata.

    No annotation panel — just the raw game render. The companion JSON
    contains all metadata the frontend needs to display overlays.

    Args:
        npz_path: Path to .npz with rgb_frames.
        output_path: Path for the output .mp4 file.
        fps: Video frame rate (default 50, matching Box2D physics).
        variant: Agent variant name (for metadata). E.g. "labeled", "blind".
        condition: Condition name (for metadata). E.g. "full-variation-easy".

    Returns:
        Tuple of (mp4_path, json_path).
    """
    import imageio.v3 as iio
    from lunar_lander.src.episode_io import load_episode

    npz_path = Path(npz_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ep = load_episode(npz_path)
    if ep["rgb_frames"] is None:
        raise ValueError(f"{npz_path} has no rgb_frames")

    frames = ep["rgb_frames"]

    # Write mp4 — raw game frames, no annotation.
    iio.imwrite(
        str(output_path),
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )

    # Compute behavioral metrics from the episode data.
    actions = ep["actions"]
    behavioral = {}
    if len(actions) > 1:
        main_thrust = actions[:, 0]
        behavioral["thrust_autocorr_lag1"] = (
            float(np.corrcoef(main_thrust[:-1], main_thrust[1:])[0, 1])
            if len(main_thrust) > 2
            else 0.0
        )
        behavioral["thrust_duty_cycle"] = float(np.mean(main_thrust > 0.05))
        behavioral["fuel_efficiency"] = float(
            ep["metadata"].get("total_reward", 0)
        ) / max(float(np.sum(np.abs(actions))), 1e-6)
    else:
        behavioral["thrust_autocorr_lag1"] = 0.0
        behavioral["thrust_duty_cycle"] = 0.0
        behavioral["fuel_efficiency"] = 0.0

    # Build companion JSON.
    duration = len(frames) / fps
    companion = {
        "physics_config": ep["metadata"].get("physics_config", {}),
        "outcome": ep["metadata"].get("outcome", "unknown"),
        "seed": ep["metadata"].get("seed", 0),
        "n_steps": len(actions),
        "total_reward": ep["metadata"].get("total_reward", 0),
        "duration_seconds": round(duration, 2),
        "behavioral_metrics": behavioral,
        "variant": variant,
        "condition": condition,
        "source_npz": str(npz_path),
    }

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(companion, f, indent=2)

    return output_path, json_path
