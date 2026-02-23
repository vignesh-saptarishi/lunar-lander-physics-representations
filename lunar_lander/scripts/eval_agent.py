#!/usr/bin/env python
"""Evaluate a trained Lunar Lander RL agent.

Loads a checkpoint, reads variant/algo from config.json (saved at training
time), runs episodes with domain-randomized physics, prints performance
stats with per-physics-regime breakdown, saves structured output (JSON + CSV),
generates diagnostic plots, and optionally records annotated mp4 videos.

Uses evaluate_agent() from eval_utils.py — the same evaluation code used
by OutcomeEvalCallback during training. Ground truth outcomes (landed,
crashed, out_of_bounds) come from the env's info dict, not heuristic
reward thresholds.

Usage:
    # Basic eval (50 episodes, print stats)
    python lunar_lander/scripts/eval_agent.py \
        --checkpoint-dir runs/lunar_lander/blind-ppo-easy

    # With plots, structured output, and videos — all in one dir
    python lunar_lander/scripts/eval_agent.py \
        --checkpoint-dir runs/lunar_lander/blind-ppo-easy \
        --episodes 100 --output-dir /tmp/eval \
        --record-videos --n-videos 5

    # Multi-profile evaluation (per-profile breakdown + comparison plots)
    python lunar_lander/scripts/eval_agent.py \
        --checkpoint-dir runs/lunar_lander/blind-ppo-easy \
        --profiles easy,medium,hard --episodes 50 --output-dir /tmp/eval

    # Parallel evaluation (8 envs)
    python lunar_lander/scripts/eval_agent.py \
        --checkpoint-dir runs/lunar_lander/blind-ppo-easy \
        --episodes 200 --n-envs 8

    # Evaluate a specific intermediate checkpoint
    python lunar_lander/scripts/eval_agent.py \
        --checkpoint-dir runs/lunar_lander/blind-ppo-easy \
        --model rl_model_1000000_steps.zip
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, REPO_ROOT)

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig
from lunar_lander.src.eval_utils import (
    evaluate_agent,
    compute_summary,
    plot_eval_summary,
    resolve_model_path,
    resolve_vec_normalize_path,
    load_training_config,
    make_env_factory,
    build_eval_batches,
    _make_env_thunk,
)
from lunar_lander.src.episode_io import save_episode
from lunar_lander.scripts.visualize_trajectory import visualize_episode

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _record_annotated_videos(
    model,
    env_fn,
    video_dir,
    n_episodes=5,
    seed=0,
    vec_normalize_path=None,
    deterministic=True,
    fps=50,
):
    """Record episodes as annotated mp4 videos.

    Runs the agent through the full VecEnv + VecNormalize stack (so obs
    are correctly normalized and the agent behaves as trained), but reaches
    through to the unwrapped ParameterizedLunarLander each step to grab:
      - The full 15D raw state vector (for annotation panel)
      - The RGB frame (for the game view)

    Each episode is saved as .npz via episode_io, then rendered to mp4
    via visualize_trajectory's annotation pipeline.

    Returns:
        List of dicts: {'path': mp4_path, 'reward': float, 'steps': int}
    """
    os.makedirs(video_dir, exist_ok=True)
    npz_dir = os.path.join(video_dir, "_npz")
    os.makedirs(npz_dir, exist_ok=True)

    # Build single-env VecEnv with normalization (same as run_episodes)
    vec_env = DummyVecEnv([_make_env_thunk(env_fn, seed)])
    if vec_normalize_path is not None:
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Reach through to the unwrapped base env for raw state + frames.
    # Stack: VecNormalize -> DummyVecEnv -> [wrappers] -> ParameterizedLunarLander
    inner = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    base_env = inner.envs[0].unwrapped

    # Enable rendering so render() produces rgb_array frames
    base_env.render_mode = "rgb_array"

    results = []
    obs = vec_env.reset()

    # The base env stores _last_obs (15D raw state) after every step/reset.
    # This is the full state before any wrapper transforms — exactly what
    # the annotation panel needs.

    # Episode tracking
    ep_states = [base_env._last_obs.copy()]
    ep_actions = []
    ep_rewards = []
    ep_dones = []
    ep_frames = [base_env.render()]
    ep_reward_total = 0.0

    while len(results) < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, infos = vec_env.step(action)

        ep_reward_total += float(reward[0])
        ep_states.append(base_env._last_obs.copy())
        ep_actions.append(action[0].copy())
        ep_rewards.append(float(reward[0]))
        ep_dones.append(bool(done[0]))
        ep_frames.append(base_env.render())

        if done[0]:
            ep_idx = len(results)

            # Classify outcome from final step reward
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
                "terrain_segments": [list(seg) for seg in base_env.terrain_segments],
            }

            # Save .npz
            npz_path = os.path.join(npz_dir, f"episode_{ep_idx:03d}.npz")
            save_episode(
                path=npz_path,
                states=np.array(ep_states, dtype=np.float32),
                actions=np.array(ep_actions, dtype=np.float32),
                rewards=np.array(ep_rewards, dtype=np.float32),
                dones=np.array(ep_dones, dtype=bool),
                metadata=metadata,
                rgb_frames=np.array(ep_frames, dtype=np.uint8),
            )

            # Render annotated mp4
            mp4_path = os.path.join(video_dir, f"episode_{ep_idx:03d}.mp4")
            visualize_episode(npz_path, mp4_path, scale=1, fps=fps)

            results.append(
                {
                    "path": mp4_path,
                    "reward": ep_reward_total,
                    "steps": len(ep_actions),
                    "outcome": outcome,
                }
            )

            # Reset for next episode
            ep_states = [base_env._last_obs.copy()]
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            ep_frames = [base_env.render()]
            ep_reward_total = 0.0

    vec_env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Lunar Lander RL agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory with model.zip, config.json, vec_normalize.pkl",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model file (default: model.zip). " "Filename or full path.",
    )
    parser.add_argument(
        "--episodes", type=int, default=50, help="Number of eval episodes (default: 50)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Parallel env workers (default: 1)"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help="Comma-separated profile names (e.g. 'easy,medium,hard')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Save results (JSON + CSV + plots + videos) here",
    )
    parser.add_argument(
        "--record-videos", action="store_true", help="Record annotated mp4 videos"
    )
    parser.add_argument(
        "--n-videos", type=int, default=5, help="Number of video episodes (default: 5)"
    )

    args = parser.parse_args()

    # --- Load agent using shared helpers from eval_utils ---
    try:
        model_path = resolve_model_path(args.checkpoint_dir, args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    vec_norm_path = resolve_vec_normalize_path(args.checkpoint_dir, model_path)
    if vec_norm_path is None:
        print("  (no matching vec_normalize found — running without obs normalization)")

    try:
        train_config = load_training_config(args.checkpoint_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    variant = train_config["variant"]
    algo = train_config.get("algo", "ppo")
    history_k = train_config.get("history_k", 8)
    n_rays = train_config.get("n_rays", 7)

    print(
        f"Loaded config: {variant} / {algo.upper()} / "
        f"{train_config.get('total_steps', '?'):,} steps"
    )

    env_fn = make_env_factory(variant=variant, n_rays=n_rays, history_k=history_k)

    from stable_baselines3 import PPO, SAC

    print(f"\nLoading model from {model_path}")
    AlgoClass = PPO if algo == "ppo" else SAC
    model = AlgoClass.load(model_path, device="auto")

    eval_batches = build_eval_batches(
        variant=variant,
        n_rays=n_rays,
        history_k=history_k,
        profiles_str=args.profiles,
        default_env_fn=env_fn,
    )

    # Run evaluation for each profile batch.
    all_episodes = []
    per_profile_summaries = {}
    for prof_name, batch_env_fn in eval_batches:
        print(f"\nEvaluating with profile: {prof_name} ({args.episodes} episodes)...")
        result = evaluate_agent(
            model=model,
            env_fn=batch_env_fn,
            n_episodes=args.episodes,
            seed=args.seed,
            vec_normalize_path=vec_norm_path,
            deterministic=True,
            n_envs=args.n_envs,
            profile=prof_name,
        )
        all_episodes.extend(result["episodes"])
        per_profile_summaries[prof_name] = result["summary"]

    # Compute overall summary across all profiles.
    overall_summary = compute_summary(all_episodes)

    # --- Print results ---
    print("\n" + "=" * 60)
    print(f"Evaluation: generalist-{variant} / {algo.upper()}")
    print("=" * 60)
    print(f"  Episodes:       {overall_summary['n_episodes']}")
    print(
        f"  Mean reward:    {overall_summary['mean_reward']:>8.1f} "
        f"+/- {overall_summary['std_reward']:.1f}"
    )
    print(f"  Mean steps:     {overall_summary['mean_steps']:>8.1f}")
    print()
    print(f"  Outcomes:")
    print(
        f"    Landed:       {overall_summary['n_landed']:>3d}/"
        f"{overall_summary['n_episodes']} "
        f"({overall_summary['landed_pct']:.0f}%)"
    )
    print(
        f"    Crashed:      {overall_summary['n_crashed']:>3d}/"
        f"{overall_summary['n_episodes']} "
        f"({overall_summary['crashed_pct']:.0f}%)"
    )
    print(
        f"    Out of bounds:{overall_summary['n_out_of_bounds']:>3d}/"
        f"{overall_summary['n_episodes']} "
        f"({overall_summary['out_of_bounds_pct']:.0f}%)"
    )
    print(
        f"    Timeout:      {overall_summary['n_timeout']:>3d}/"
        f"{overall_summary['n_episodes']} "
        f"({overall_summary['timeout_pct']:.0f}%)"
    )

    # Per-profile breakdown (if multiple profiles evaluated)
    if len(per_profile_summaries) > 1:
        print(f"\n  Per-profile breakdown:")
        for prof_name, s in per_profile_summaries.items():
            print(
                f"    {prof_name:12s}: {s['n_landed']}/{s['n_episodes']} landed "
                f"({s['landed_pct']:.0f}%), "
                f"reward {s['mean_reward']:.1f} +/- {s['std_reward']:.1f}"
            )

    # Physics correlations (landed vs crashed)
    landed_eps = [e for e in all_episodes if e["outcome"] == "landed"]
    crashed_eps = [e for e in all_episodes if e["outcome"] == "crashed"]
    if landed_eps and crashed_eps:
        print(f"\n  Physics correlations (landed vs crashed):")
        for pname in list(LunarLanderPhysicsConfig.PARAM_NAMES) + ["twr"]:
            l_vals = [e[pname] for e in landed_eps if e.get(pname) is not None]
            c_vals = [e[pname] for e in crashed_eps if e.get(pname) is not None]
            if l_vals and c_vals:
                print(
                    f"    {pname:>22s}: "
                    f"landed={np.mean(l_vals):.1f}\u00b1{np.std(l_vals):.1f}, "
                    f"crashed={np.mean(c_vals):.1f}\u00b1{np.std(c_vals):.1f}"
                )

    # --- Save structured output ---
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # JSON summary
        json_path = os.path.join(args.output_dir, "eval_results.json")
        json_data = {
            "variant": variant,
            "algo": algo,
            "n_episodes": overall_summary["n_episodes"],
            "overall": overall_summary,
            "per_profile": per_profile_summaries,
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # CSV per-episode records
        csv_path = os.path.join(args.output_dir, "eval_episodes.csv")
        if all_episodes:
            fieldnames = list(all_episodes[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_episodes)

        # Generate diagnostic plots
        plot_eval_summary(
            all_episodes,
            args.output_dir,
            per_profile_summaries=(
                per_profile_summaries if len(per_profile_summaries) > 1 else None
            ),
        )

        print(f"\n  Results saved to {args.output_dir}/")
        print(f"    {os.path.basename(json_path)}")
        print(f"    {os.path.basename(csv_path)}")

    # --- Record annotated videos ---
    if args.record_videos:
        if args.output_dir:
            video_dir = os.path.join(args.output_dir, "videos")
        else:
            video_dir = os.path.join(args.checkpoint_dir, "eval_videos")
        print(f"\nRecording {args.n_videos} annotated episodes as mp4...")

        # Use the first profile's env_fn for videos so physics match
        # what was evaluated (not bare env_fn which has no profile).
        video_env_fn = eval_batches[0][1] if args.profiles else env_fn

        video_results = _record_annotated_videos(
            model=model,
            env_fn=video_env_fn,
            video_dir=video_dir,
            n_episodes=args.n_videos,
            seed=args.seed + 10000,  # different seed from eval
            vec_normalize_path=vec_norm_path,
            deterministic=True,
        )

        print(f"\n  Videos saved to {video_dir}/")
        for vr in video_results:
            print(
                f"    {os.path.basename(vr['path']):25s}  "
                f"reward={vr['reward']:>7.1f}  steps={vr['steps']}  {vr['outcome']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
