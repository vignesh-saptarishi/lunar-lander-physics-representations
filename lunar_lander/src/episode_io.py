"""Episode save/load for Lunar Lander trajectories.

Defines the .npz file format for storing episodes and provides
save/load functions. This is the canonical episode format for
the Lunar Lander testbed — all collection scripts, visualization,
and data loaders use this module.

File format (.npz):
    states:        (T+1, 15) float32 — state at each timestep (T+1 because
                   includes initial state from reset, before any action)
    actions:       (T, 2) float32 — continuous actions (main_thrust, side_thrust)
    rewards:       (T,) float32 — reward at each step
    dones:         (T,) bool — terminated flag at each step
    rgb_frames:    (T+1, H, W, 3) uint8 — optional, only if save_frames=True
    metadata_json: str — JSON-serialized dict with physics_config, outcome, etc.

Convention: T is the number of actions taken. states has T+1 entries
(initial + one per step). rgb_frames, if present, also has T+1 entries
(one per state). actions/rewards/dones have T entries (one per step).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lunar_lander.src.physics_config import LunarLanderPhysicsConfig


def save_episode(
    path: str | Path,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    metadata: dict,
    rgb_frames: np.ndarray | None = None,
) -> Path:
    """Save an episode to .npz format.

    Args:
        path: Output file path (will add .npz suffix if missing).
        states: (T+1, 15) float32 state array. Includes initial state
            from reset() plus one state per step.
        actions: (T, 2) float32 action array. One action per step.
        rewards: (T,) float32 reward array.
        dones: (T,) bool termination flags.
        metadata: Dict with episode info. Should contain at minimum:
            - physics_config: dict from LunarLanderPhysicsConfig.to_dict()
            - outcome: str ("landed", "crashed", "timeout", "out_of_bounds")
            - seed: int
            Additional keys are preserved (calibration, etc.).
        rgb_frames: Optional (T+1, H, W, 3) uint8 frame array.
            One frame per state (including initial).

    Returns:
        Path to the saved .npz file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate shapes for consistency.
    n_steps = len(actions)
    assert states.shape[0] == n_steps + 1, (
        f"states has {states.shape[0]} entries but expected {n_steps + 1} "
        f"(actions has {n_steps} steps)"
    )
    assert rewards.shape == (n_steps,), f"rewards shape {rewards.shape} != ({n_steps},)"
    assert dones.shape == (n_steps,), f"dones shape {dones.shape} != ({n_steps},)"

    # Build save dict — metadata is JSON-serialized to a string.
    save_dict = {
        "states": states.astype(np.float32),
        "actions": actions.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "dones": dones.astype(bool),
        "metadata_json": json.dumps(metadata),
    }

    if rgb_frames is not None:
        assert (
            rgb_frames.shape[0] == n_steps + 1
        ), f"rgb_frames has {rgb_frames.shape[0]} entries but expected {n_steps + 1}"
        save_dict["rgb_frames"] = rgb_frames.astype(np.uint8)

    np.savez_compressed(str(path), **save_dict)
    return path


def load_episode(path: str | Path) -> dict:
    """Load an episode from .npz format.

    Args:
        path: Path to the .npz file.

    Returns:
        Dict with keys:
            states: (T+1, 15) float32
            actions: (T, 2) float32
            rewards: (T,) float32
            dones: (T,) bool
            metadata: dict (parsed from JSON)
            rgb_frames: (T+1, H, W, 3) uint8 or None
            physics_config: LunarLanderPhysicsConfig (convenience, from metadata)
    """
    data = np.load(str(path), allow_pickle=False)

    metadata = json.loads(str(data["metadata_json"]))

    result = {
        "states": data["states"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "dones": data["dones"],
        "metadata": metadata,
        "rgb_frames": data["rgb_frames"] if "rgb_frames" in data else None,
    }

    # Convenience: parse physics config from metadata if present.
    if "physics_config" in metadata:
        result["physics_config"] = LunarLanderPhysicsConfig.from_dict(
            metadata["physics_config"]
        )

    return result


def run_episode(
    env,
    policy_fn,
    seed: int = 42,
    max_steps: int = 300,
    save_frames: bool = False,
) -> dict:
    """Run a single episode and collect trajectory data.

    Convenience function that runs a policy in the env, collects all
    data in the episode format, and returns it ready for save_episode().

    Args:
        env: ParameterizedLunarLander instance. Must already be created
            with the desired physics_config. If save_frames=True, env
            must have render_mode="rgb_array".
        policy_fn: Callable(obs) -> action. Takes (15,) obs, returns (2,) action.
        seed: RNG seed for env.reset().
        max_steps: Maximum steps before timeout.
        save_frames: If True, capture rgb_array frames each step.

    Returns:
        Dict with:
            states, actions, rewards, dones: numpy arrays
            rgb_frames: numpy array or None
            metadata: dict with physics_config, outcome, seed, n_steps
    """
    obs, info = env.reset(seed=seed)

    states_list = [obs.copy()]
    actions_list = []
    rewards_list = []
    dones_list = []
    frames_list = []

    if save_frames:
        frame = env.render()
        assert frame is not None, (
            "save_frames=True but env.render() returned None. "
            "Create env with render_mode='rgb_array'."
        )
        frames_list.append(frame)

    # Track episode outcome for metadata.
    outcome = "timeout"
    total_reward = 0.0

    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)

        states_list.append(obs.copy())
        actions_list.append(action.copy())
        rewards_list.append(reward)
        dones_list.append(terminated)
        total_reward += reward

        if save_frames:
            frame = env.render()
            frames_list.append(frame)

        if terminated:
            # Classify outcome from reward signal.
            # +100 = landed (lander came to rest), -100 = crashed or OOB.
            if reward >= 100:
                outcome = "landed"
            else:
                # Distinguish crash from out-of-bounds using state.
                # obs[0] is normalized x position; |x| >= 1 means OOB.
                if abs(obs[0]) >= 1.0:
                    outcome = "out_of_bounds"
                else:
                    outcome = "crashed"
            break

    metadata = {
        "physics_config": info["physics_config"],
        "terrain_segments": info.get("terrain_segments", []),
        "outcome": outcome,
        "seed": seed,
        "n_steps": len(actions_list),
        "total_reward": float(total_reward),
    }

    result = {
        "states": np.array(states_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.float32),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=bool),
        "metadata": metadata,
        "rgb_frames": np.array(frames_list, dtype=np.uint8) if save_frames else None,
    }

    return result
