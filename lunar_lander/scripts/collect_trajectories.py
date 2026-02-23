#!/usr/bin/env python
"""Collect trajectories from a trained Lunar Lander agent.

Loads a trained checkpoint, runs N episodes through the full
VecNormalize + wrapper stack, and saves complete state/action/reward
traces as .npz files. No RGB frames, no metrics — just raw trajectory data.

For metrics computation, pipe the output directory to compute_metrics.py.
For the full pipeline (collect + metrics), use run_eval_pipeline.py.

Usage:
    # Single agent, 50 episodes
    python lunar_lander/scripts/collect_trajectories.py \
        --checkpoint-dir /path/to/agent --episodes 50

    # Specific intermediate checkpoint
    python lunar_lander/scripts/collect_trajectories.py \
        --checkpoint-dir /path/to/agent --model rl_model_2000000_steps.zip

    # Batch: all agents under a parent directory
    python lunar_lander/scripts/collect_trajectories.py \
        --agents-dir /path/to/rl_agents/ --episodes 50

    # Custom output location
    python lunar_lander/scripts/collect_trajectories.py \
        --checkpoint-dir /path/to/agent --episodes 50 \
        --output-dir /tmp/trajectories
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.eval_utils import (
    resolve_model_path,
    resolve_vec_normalize_path,
    load_training_config,
    make_env_factory,
    build_eval_batches,
    _make_env_thunk,
)
from lunar_lander.src.episode_io import save_episode

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _check_existing_npz(output_dir: str) -> int:
    """Count existing .npz files in a directory. Returns 0 if dir doesn't exist."""
    if not os.path.isdir(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith(".npz")])


def _collect_episodes(
    model,
    env_fn,
    output_dir,
    n_episodes,
    seed=0,
    vec_normalize_path=None,
    deterministic=True,
    profile=None,
    save_frames=False,
):
    """Collect full trajectory .npz files from a trained agent.

    Runs episodes through the VecNormalize + wrapper stack (so the agent
    gets correctly normalized observations), but reaches through to the
    base ParameterizedLunarLander to capture:
      - Raw 15D state vectors (_last_obs) at each timestep
      - Actions (from model.predict)
      - Rewards, dones
      - Physics config + outcome metadata

    Unlike evaluate_agent() which only returns summary metrics, this
    saves complete trajectories for downstream analysis (trajectory
    metrics, state-space coverage, action distributions).

    When save_frames=True, captures rgb_array frames at each timestep and
    includes them in the .npz. Requires env to have render_mode="rgb_array".
    Files grow from ~20KB to ~1-2MB per episode.

    Args:
        model: Trained SB3 model (or any object with .predict(obs) -> (action, _)).
        env_fn: Factory taking seed, returns wrapped env.
        output_dir: Directory for .npz files.
        n_episodes: Number of episodes to collect.
        seed: Base seed for env creation.
        vec_normalize_path: Path to VecNormalize .pkl stats.
        deterministic: Use deterministic policy (no exploration noise).
        profile: Profile name string for metadata tagging.

    Returns:
        List of per-episode summary dicts with keys:
            npz_path, outcome, reward, steps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build single-env VecEnv with normalization — same pattern as
    # _record_annotated_videos in eval_agent.py. We need the VecEnv
    # stack so the agent gets the same normalized observations it saw
    # during training.
    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        # Eval mode: don't update running stats, don't normalize rewards
        vec_env.training = False
        vec_env.norm_reward = False

    # Reach through the wrapper stack to the base ParameterizedLunarLander.
    # We need _last_obs (raw 15D state) and _physics_config from the
    # unwrapped env, since wrappers transform/mask the observation.
    inner = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    base_env = inner.envs[0].unwrapped

    results = []
    obs = vec_env.reset()

    # Per-episode accumulators
    ep_states = [base_env._last_obs.copy()]
    ep_actions = []
    ep_rewards = []
    ep_dones = []
    ep_frames = [base_env.render()] if save_frames else []
    ep_reward_total = 0.0

    while len(results) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = vec_env.step(action)

        ep_reward_total += float(reward[0])
        ep_states.append(base_env._last_obs.copy())
        ep_actions.append(action[0].copy())
        ep_rewards.append(float(reward[0]))
        ep_dones.append(bool(done[0]))
        if save_frames:
            ep_frames.append(base_env.render())

        if done[0]:
            ep_idx = len(results)

            # Classify outcome from final step reward.
            # +100 = landed, -100 = crashed, else timeout.
            final_reward = ep_rewards[-1]
            if final_reward >= 100:
                outcome = "landed"
            elif final_reward <= -100:
                outcome = "crashed"
            else:
                outcome = "timeout"

            # Build metadata matching episode_io format
            physics_config = base_env._physics_config.to_dict()
            metadata = {
                "physics_config": physics_config,
                "outcome": outcome,
                "seed": seed + ep_idx,
                "episode_length": len(ep_actions),
                "total_reward": ep_reward_total,
                "policy": "rl_agent",
                "profile": profile,
            }

            npz_path = os.path.join(output_dir, f"episode_{ep_idx:04d}.npz")
            save_episode(
                path=npz_path,
                states=np.array(ep_states, dtype=np.float32),
                actions=np.array(ep_actions, dtype=np.float32),
                rewards=np.array(ep_rewards, dtype=np.float32),
                dones=np.array(ep_dones, dtype=bool),
                metadata=metadata,
                rgb_frames=np.array(ep_frames, dtype=np.uint8) if save_frames else None,
            )

            results.append(
                {
                    "npz_path": npz_path,
                    "outcome": outcome,
                    "reward": ep_reward_total,
                    "steps": len(ep_actions),
                }
            )

            if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
                print(f"  Collected {ep_idx + 1}/{n_episodes} episodes")

            # Reset accumulators for next episode
            ep_states = [base_env._last_obs.copy()]
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            ep_frames = [base_env.render()] if save_frames else []
            ep_reward_total = 0.0

    vec_env.close()
    return results


def _find_agent_dirs(parent_dir):
    """Find all agent checkpoint directories under a parent directory.

    Walks the directory tree looking for subdirs that contain either
    model.zip or config.json — indicators of a training checkpoint.
    """
    agent_dirs = []
    for root, dirs, files in os.walk(parent_dir, followlinks=True):
        if "model.zip" in files or "config.json" in files:
            agent_dirs.append(root)
            # Don't recurse into this dir's subdirs (they're checkpoints, not agents)
            dirs.clear()
    return sorted(agent_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectories from trained Lunar Lander agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint-dir", help="Single agent checkpoint directory")
    group.add_argument(
        "--agents-dir", help="Parent dir — collect from all agents underneath"
    )

    parser.add_argument(
        "--model", default=None, help="Specific model file (default: model.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Episodes per agent (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base random seed (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {checkpoint-dir}/trajectories/)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="Comma-separated profile names (e.g. 'easy,medium')",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing trajectories (deletes old .npz files first)",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        choices=["zero", "shuffle", "mean", "noise"],
        help="Label corruption type for eval-time manipulation. "
        "Only meaningful for labeled agents.",
    )
    parser.add_argument(
        "--corruption-sigma",
        type=float,
        default=0.1,
        help="Noise std as fraction of param range (default: 0.1). "
        "Only used with --corruption noise.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Capture RGB frames at each timestep and include in .npz files. "
        "Increases file size from ~20KB to ~1-2MB per episode.",
    )
    parser.add_argument(
        "--corruption-means-dir",
        type=str,
        default=None,
        help="Directory of .npz files to compute training means from. "
        "Required for --corruption mean. Defaults to the agent's "
        "own trajectories/ dir if it exists.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir:
        agent_dirs = [args.checkpoint_dir]
    else:
        agent_dirs = _find_agent_dirs(args.agents_dir)
        if not agent_dirs:
            print(f"ERROR: No agent checkpoints found under {args.agents_dir}")
            sys.exit(1)
        print(f"Found {len(agent_dirs)} agents under {args.agents_dir}")

    # Collect from each agent
    all_summaries = []
    for agent_dir in agent_dirs:
        agent_name = os.path.basename(agent_dir)
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"{'='*60}")

        # Load config and model
        try:
            model_path = resolve_model_path(agent_dir, args.model)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        vec_norm_path = resolve_vec_normalize_path(agent_dir, model_path)

        try:
            train_config = load_training_config(agent_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        variant = train_config["variant"]
        algo = train_config.get("algo", "ppo")
        history_k = train_config.get("history_k", 8)
        n_rays = train_config.get("n_rays", 7)

        print(f"  Config: {variant} / {algo.upper()}")

        # Load model
        from stable_baselines3 import PPO, SAC

        AlgoClass = PPO if algo == "ppo" else SAC
        model = AlgoClass.load(model_path, device="auto")

        # Auto-detect training profile from config.json unless --profiles overrides.
        effective_profiles = args.profiles
        if not effective_profiles:
            train_profile = train_config.get("profile")
            if train_profile:
                effective_profiles = train_profile
                print(f"  Auto-detected training profile: {train_profile}")

        # Build env factory and eval batches
        render_mode = "rgb_array" if args.save_frames else None
        env_fn = make_env_factory(
            variant=variant, n_rays=n_rays, history_k=history_k, render_mode=render_mode
        )
        eval_batches = build_eval_batches(
            variant=variant,
            n_rays=n_rays,
            history_k=history_k,
            profiles_str=effective_profiles,
            default_env_fn=env_fn,
            render_mode=render_mode,
        )

        # Apply label corruption if requested.
        # Only meaningful for labeled agents — blind agents have physics dims
        # stripped (dims 8-14 are rays, not physics), and history agents stack
        # the already-processed blind obs. Corrupting those would corrupt the
        # wrong dimensions.
        if args.corruption:
            if variant != "labeled":
                print(
                    f"  SKIP: --corruption only applies to labeled agents "
                    f"(this is '{variant}')"
                )
                continue

            training_means = None
            if args.corruption == "mean":
                from lunar_lander.src.label_corruption import compute_training_means

                means_dir = args.corruption_means_dir
                if means_dir is None:
                    # Default: use this agent's own trajectories
                    means_dir = os.path.join(agent_dir, "trajectories")
                if not os.path.isdir(means_dir):
                    print(
                        f"  SKIP: --corruption mean requires trajectory dir at {means_dir}"
                    )
                    continue
                print(f"  Computing training means from {means_dir}")
                training_means = compute_training_means(means_dir)

            # Wrap each batch's env_fn with corruption
            from lunar_lander.src.eval_utils import wrap_env_fn_with_corruption

            eval_batches = [
                (
                    prof_name,
                    wrap_env_fn_with_corruption(
                        batch_env_fn,
                        corruption_type=args.corruption,
                        corruption_sigma=args.corruption_sigma,
                        training_means=training_means,
                    ),
                )
                for prof_name, batch_env_fn in eval_batches
            ]
            print(
                f"  Corruption: {args.corruption}"
                + (
                    f" (sigma={args.corruption_sigma})"
                    if args.corruption == "noise"
                    else ""
                )
            )

        # Collect for each profile.
        # In batch mode with --output-dir, use per-agent subdirs to avoid overwrites.
        if args.output_dir and len(agent_dirs) > 1:
            output_base = os.path.join(args.output_dir, agent_name)
        else:
            output_base = args.output_dir or os.path.join(agent_dir, "trajectories")

        # Corruption runs save to a separate subdir to avoid overwriting baselines.
        # e.g. trajectories-zero/, trajectories-noise-s0.1/
        if args.corruption:
            corruption_tag = args.corruption
            if args.corruption == "noise":
                corruption_tag = f"noise-s{args.corruption_sigma}"
            if output_base.endswith("trajectories"):
                output_base = (
                    output_base[: -len("trajectories")]
                    + f"trajectories-{corruption_tag}"
                )
            else:
                output_base = f"{output_base}-{corruption_tag}"

        for prof_name, batch_env_fn in eval_batches:
            if len(eval_batches) > 1:
                collect_dir = os.path.join(output_base, prof_name)
            else:
                collect_dir = output_base

            # Check for existing trajectories before collecting.
            existing = _check_existing_npz(collect_dir)
            if existing and not args.force:
                print(f"\n  STOP: {existing} existing .npz files in {collect_dir}/")
                print(
                    f"  Use --force to overwrite, or run_eval_pipeline.py --skip-collect to reuse them."
                )
                continue
            if existing and args.force:
                import glob as globmod

                old_files = globmod.glob(os.path.join(collect_dir, "*.npz"))
                for f in old_files:
                    os.remove(f)
                print(f"\n  Removed {len(old_files)} old .npz files (--force)")

            print(f"\n  Collecting {args.episodes} episodes (profile: {prof_name})...")
            results = _collect_episodes(
                model=model,
                env_fn=batch_env_fn,
                output_dir=collect_dir,
                n_episodes=args.episodes,
                seed=args.seed,
                vec_normalize_path=vec_norm_path,
                deterministic=args.deterministic,
                profile=prof_name,
                save_frames=args.save_frames,
            )

            n_landed = sum(1 for r in results if r["outcome"] == "landed")
            mean_reward = np.mean([r["reward"] for r in results])
            print(f"  Saved {len(results)} episodes to {collect_dir}/")
            print(
                f"  Landed: {n_landed}/{len(results)} ({100*n_landed/len(results):.0f}%), "
                f"mean reward: {mean_reward:.1f}"
            )

            all_summaries.append(
                {
                    "agent": agent_name,
                    "profile": prof_name,
                    "episodes": len(results),
                    "landed": n_landed,
                    "landed_pct": 100 * n_landed / len(results),
                    "mean_reward": mean_reward,
                }
            )

        # Save collection metadata
        meta = {
            "checkpoint_dir": agent_dir,
            "variant": variant,
            "algo": algo,
            "profiles": [p for p, _ in eval_batches],
            "episodes_per_profile": args.episodes,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "corruption": args.corruption,
            "corruption_sigma": (
                args.corruption_sigma if args.corruption == "noise" else None
            ),
        }
        meta_path = os.path.join(output_base, "collection_meta.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Print summary table for batch mode
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("Collection Summary")
        print(f"{'='*70}")
        print(f"{'Agent':<40s} {'Profile':<10s} {'Landed%':>8s} {'Reward':>8s}")
        print("-" * 70)
        for s in all_summaries:
            print(
                f"{s['agent']:<40s} {s['profile']:<10s} "
                f"{s['landed_pct']:>7.0f}% {s['mean_reward']:>8.1f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
