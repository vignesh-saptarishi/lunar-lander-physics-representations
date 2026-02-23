#!/usr/bin/env python3
"""Visualize Lunar Lander episodes as annotated MP4 videos.

Takes .npz episode files (from episode_io.save_episode) and renders
annotated frames: game view + right-side panel with physics config,
lander state, actions, rewards, and episode outcome.

Matches the platformer's visualization style (PIL annotations, imageio MP4).

Usage:
    # Single episode
    python lunar_lander/scripts/visualize_trajectory.py episode.npz

    # Custom output path and scale
    python lunar_lander/scripts/visualize_trajectory.py episode.npz -o my_video.mp4 --scale 2

    # Directory of episodes (sample N)
    python lunar_lander/scripts/visualize_trajectory.py data/episodes/ -n 5

    # Skip annotations (raw frames only)
    python lunar_lander/scripts/visualize_trajectory.py episode.npz --no-annotations
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- State vector index mapping ---
# Base state (normalized coords from Gymnasium):
#   [0] x_pos, [1] y_pos, [2] x_vel, [3] y_vel,
#   [4] angle, [5] ang_vel, [6] left_leg, [7] right_leg
# Physics params (raw values):
#   [8] gravity, [9] main_engine_power, [10] side_engine_power,
#   [11] lander_density, [12] angular_damping, [13] wind_power,
#   [14] turbulence_power

_PHYSICS_NAMES = [
    ("gravity", "g"),
    ("main_power", "thrust"),
    ("side_power", "side"),
    ("density", "dens"),
    ("ang_damp", "damp"),
    ("wind", "wind"),
    ("turbulence", "turb"),
]


def _get_font(size=12):
    """Get a monospace font, falling back gracefully."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _action_bar(value, width=10):
    """Create a text bar visualization for an action value in [-1, 1].

    Returns a string like '=====>    ' (positive) or '    <=====' (negative).
    """
    n = int(abs(value) * width)
    if value > 0.05:
        return "=" * n + ">"
    elif value < -0.05:
        return "<" + "=" * n
    else:
        return "-"


def _outcome_color(outcome):
    """Color for episode outcome text."""
    if outcome == "landed":
        return (100, 255, 100)  # green
    elif outcome == "crashed":
        return (255, 100, 100)  # red
    elif outcome == "out_of_bounds":
        return (255, 180, 50)  # orange
    else:
        return (200, 200, 200)  # gray (timeout)


def render_annotated_frame(
    rgb,
    state,
    action,
    reward,
    cumulative_reward,
    step,
    total_steps,
    metadata,
    scale=2,
    annotate=True,
):
    """Render a single annotated frame.

    Args:
        rgb: (H, W, 3) uint8 game frame.
        state: (15,) float32 state vector.
        action: (2,) float32 action vector, or None for initial frame.
        reward: float, step reward (0 for initial frame).
        cumulative_reward: float, cumulative reward so far.
        step: int, current step index.
        total_steps: int, total steps in episode.
        metadata: dict with physics_config, outcome, etc.
        scale: int, upscale factor for game frame.
        annotate: bool, whether to add text annotations.

    Returns:
        PIL Image of the annotated frame.
    """
    h, w = rgb.shape[:2]

    # LunarLander frames are 600x400 — already large enough to read.
    # Scale=1 is the default. Only upscale if the frame is small (e.g. 84x84 pixel obs).
    scaled_w = w * scale
    scaled_h = h * scale
    img = Image.fromarray(rgb).resize((scaled_w, scaled_h), Image.NEAREST)

    if not annotate:
        return img

    # --- Font sizing ---
    # Fixed readable sizes: 16pt headers, 13pt body. These work well
    # at 600x400 game frames and produce ~26 lines of annotations.
    header_size = 16
    body_size = 13
    font = _get_font(header_size)
    small_font = _get_font(body_size)
    line_h = header_size + 6  # header line height (~22px)
    small_line_h = body_size + 4  # body line height (~17px)
    section_gap = 6  # gap between sections

    # Pre-compute total text height to size the canvas properly.
    # Sections: step(1h) + physics(1h+8b) + lander(1h+5b) +
    #           action(1h+2b) + reward(1h+2b) + outcome(1h) + margins
    n_header_lines = 6  # step, physics, lander, action, reward, outcome
    n_body_lines = 17  # 7 physics + 1 TWR + 5 lander + 2 action + 2 reward
    n_gaps = 5  # gaps between sections
    text_height = (
        20  # top margin
        + n_header_lines * line_h
        + n_body_lines * small_line_h
        + n_gaps * section_gap
        + 20  # bottom margin
    )

    # Canvas height: whichever is taller — game frame or text.
    # Round up to even — libx264 requires both dimensions divisible by 2.
    canvas_h = max(scaled_h, text_height)
    canvas_h += canvas_h % 2

    # --- Build canvas with right-side annotation panel ---
    panel_w = max(300, scaled_w // 2)
    canvas_w = scaled_w + panel_w
    canvas_w += canvas_w % 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 35))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    x0 = scaled_w + 12  # left edge of annotation panel
    y = 10
    header_color = (150, 200, 255)  # light cyan for section headers
    text_color = (200, 200, 200)  # light gray for data
    dim_color = (130, 130, 140)  # dim gray for labels

    # --- Header: step counter ---
    draw.text((x0, y), f"Step {step}/{total_steps}", fill=(255, 255, 255), font=font)
    y += line_h + section_gap

    # --- Physics config ---
    draw.text((x0, y), "Physics", fill=header_color, font=font)
    y += line_h
    for i, (_, short) in enumerate(_PHYSICS_NAMES):
        val = state[8 + i]
        draw.text(
            (x0, y),
            f"  {short:>6s}: {val:+8.2f}",
            fill=text_color,
            font=small_font,
        )
        y += small_line_h
    # Derived: thrust-to-weight ratio. Computed from gravity (idx 8),
    # main_engine_power (idx 9), and lander_density (idx 11).
    # TWR = thrust * fps / (density * body_area * |gravity|)
    _gravity = abs(state[8])
    _thrust = state[9]
    _density = state[11]
    _body_area = 867.0 / 900.0  # from LunarLanderPhysicsConfig.BODY_AREA
    if _gravity > 0 and _density > 0:
        twr = _thrust * 50.0 / (_density * _body_area * _gravity)
        twr_color = (
            (100, 255, 100)
            if twr >= 5.0
            else (255, 200, 100) if twr >= 2.0 else (255, 100, 100)
        )
        draw.text(
            (x0, y), f"  {'TWR':>6s}: {twr:8.1f}", fill=twr_color, font=small_font
        )
    else:
        draw.text((x0, y), f"  {'TWR':>6s}:      N/A", fill=dim_color, font=small_font)
    y += small_line_h
    y += section_gap

    # --- Lander state ---
    draw.text((x0, y), "Lander", fill=header_color, font=font)
    y += line_h
    draw.text(
        (x0, y),
        f"  pos: ({state[0]:+.2f}, {state[1]:+.2f})",
        fill=text_color,
        font=small_font,
    )
    y += small_line_h
    draw.text(
        (x0, y),
        f"  vel: ({state[2]:+.2f}, {state[3]:+.2f})",
        fill=text_color,
        font=small_font,
    )
    y += small_line_h
    draw.text(
        (x0, y), f"  angle: {state[4]:+.3f} rad", fill=text_color, font=small_font
    )
    y += small_line_h
    draw.text((x0, y), f"  ang_v: {state[5]:+.3f}", fill=text_color, font=small_font)
    y += small_line_h

    # Leg contacts — highlight green when touching.
    left_contact = state[6] > 0.5
    right_contact = state[7] > 0.5
    left_color = (100, 255, 100) if left_contact else dim_color
    right_color = (100, 255, 100) if right_contact else dim_color
    draw.text((x0, y), "  legs: ", fill=text_color, font=small_font)
    leg_x = x0 + small_font.getlength("  legs: ")
    draw.text((leg_x, y), "L", fill=left_color, font=small_font)
    draw.text(
        (leg_x + small_font.getlength("L "), y), "R", fill=right_color, font=small_font
    )
    y += small_line_h + section_gap

    # --- Action ---
    draw.text((x0, y), "Action", fill=header_color, font=font)
    y += line_h
    if action is not None:
        main_val = action[0]
        side_val = action[1]
        main_active = main_val > 0
        main_color = (255, 200, 100) if main_active else dim_color
        draw.text(
            (x0, y),
            f"  main: {_action_bar(main_val)}",
            fill=main_color,
            font=small_font,
        )
        y += small_line_h
        side_active = abs(side_val) > 0.5
        side_color = (100, 200, 255) if side_active else dim_color
        side_dir = "R" if side_val > 0.5 else "L" if side_val < -0.5 else "-"
        draw.text(
            (x0, y),
            f"  side: {side_dir} {_action_bar(side_val)}",
            fill=side_color,
            font=small_font,
        )
        y += small_line_h
    else:
        draw.text((x0, y), "  (initial)", fill=dim_color, font=small_font)
        y += small_line_h
    y += section_gap

    # --- Reward ---
    draw.text((x0, y), "Reward", fill=header_color, font=font)
    y += line_h
    r_color = (
        (100, 255, 100)
        if reward > 0.5
        else (255, 100, 100) if reward < -0.5 else text_color
    )
    draw.text((x0, y), f"  step:  {reward:+.1f}", fill=r_color, font=small_font)
    y += small_line_h
    draw.text(
        (x0, y), f"  total: {cumulative_reward:+.1f}", fill=text_color, font=small_font
    )
    y += small_line_h + section_gap

    # --- Episode outcome (show on final frame) ---
    outcome = metadata.get("outcome")
    done = step == total_steps
    if done and outcome:
        label = outcome.upper().replace("_", " ")
        draw.text((x0, y), label, fill=_outcome_color(outcome), font=font)

    return canvas


def visualize_episode(npz_path, output_path, scale=1, fps=50, annotate=True):
    """Render a single episode .npz to annotated MP4.

    Args:
        npz_path: Path to .npz episode file.
        output_path: Path for output MP4 file.
        scale: Upscale factor for game frames.
        fps: Output video FPS (default 50 to match Box2D physics FPS).
        annotate: Whether to add text annotation panel.
    """
    import imageio.v3 as iio

    npz_path = Path(npz_path)
    data = np.load(str(npz_path), allow_pickle=False)

    if "rgb_frames" not in data:
        print(f"Error: {npz_path} has no rgb_frames (was it saved without frames?)")
        sys.exit(1)

    rgb_frames = data["rgb_frames"]
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    metadata = json.loads(str(data["metadata_json"]))

    n_frames = len(rgb_frames)
    n_steps = len(actions)  # T actions, T+1 states/frames

    print(
        f"Rendering {npz_path.name}: {n_steps} steps, "
        f"{rgb_frames.shape[1]}x{rgb_frames.shape[2]} -> {scale}x"
    )

    frames = []
    cumulative_reward = 0.0

    for i in range(n_frames):
        state = states[i]

        if i == 0:
            # Initial frame: no action/reward yet.
            action = None
            reward = 0.0
        else:
            action = actions[i - 1]
            reward = float(rewards[i - 1])
            cumulative_reward += reward

        img = render_annotated_frame(
            rgb=rgb_frames[i],
            state=state,
            action=action,
            reward=reward,
            cumulative_reward=cumulative_reward,
            step=i,
            total_steps=n_steps,
            metadata=metadata,
            scale=scale,
            annotate=annotate,
        )
        frames.append(np.array(img))

    # Hold final frame for 1.5 seconds so outcome is visible.
    if frames:
        hold_count = int(fps * 1.5)
        for _ in range(hold_count):
            frames.append(frames[-1])

    # Write MP4.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        str(output_path),
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
    duration = len(frames) / fps
    print(f"Saved: {output_path} ({len(frames)} frames, {duration:.1f}s at {fps}fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Lunar Lander episodes as annotated MP4 videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Path(s) to .npz episode file(s) or directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: <input_stem>.mp4).",
    )
    parser.add_argument(
        "-n",
        "--sample",
        type=int,
        default=None,
        help="Sample N random episodes from a directory.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Upscale factor for game frames (default: 1, no upscale).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output video FPS (default: 50, matching physics FPS).",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip text annotations (raw frames only).",
    )

    args = parser.parse_args()
    input_paths = [Path(p) for p in args.input]

    # Directory mode: find .npz files, optionally sample.
    if len(input_paths) == 1 and input_paths[0].is_dir():
        dir_path = input_paths[0]
        all_npz = sorted(dir_path.rglob("*.npz"))
        if not all_npz:
            print(f"Error: no .npz files found in {dir_path}")
            sys.exit(1)

        if args.sample:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                len(all_npz), size=min(args.sample, len(all_npz)), replace=False
            )
            all_npz = [all_npz[i] for i in sorted(indices)]

        # Output directory: use -o if provided, otherwise write next to source files
        out_dir = Path(args.output) if args.output else None

        print(f"Rendering {len(all_npz)} episodes from {dir_path}")
        for npz_path in all_npz:
            if out_dir:
                # Preserve subdirectory structure relative to input dir
                rel = npz_path.relative_to(dir_path).with_suffix(".mp4")
                out_path = out_dir / rel
            else:
                out_path = npz_path.with_suffix(".mp4")
            visualize_episode(
                npz_path,
                out_path,
                scale=args.scale,
                fps=args.fps,
                annotate=not args.no_annotations,
            )
        return

    # File mode: render each .npz file.
    for i, npz_path in enumerate(input_paths):
        if not npz_path.exists():
            print(f"Error: {npz_path} not found")
            sys.exit(1)
        output = (
            args.output
            if len(input_paths) == 1 and args.output
            else str(npz_path.with_suffix(".mp4"))
        )
        visualize_episode(
            npz_path,
            output,
            scale=args.scale,
            fps=args.fps,
            annotate=not args.no_annotations,
        )


if __name__ == "__main__":
    main()
