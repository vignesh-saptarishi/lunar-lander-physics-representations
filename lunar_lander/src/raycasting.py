"""Terrain raycasting for the Lunar Lander environment.

Casts a fan of downward-pointing rays from the lander position and
intersects them with the terrain line segments. Each ray returns a
normalized distance in [0, 1]: 0 = ground contact, 1 = nothing within
sensing range. This gives the agent ground-relative spatial awareness.

The computation is pure numpy geometry — no Box2D involved. Rays are
intersected with the ~10 terrain line segments using parametric
ray-segment intersection.

Ray fan geometry:
  - N rays spread in a downward arc of `arc_degrees` total width
  - The center ray points straight down (or lander-down in ego frame)
  - Ray 0 is the leftmost (most negative angle offset), ray N-1 is rightmost
  - Angles are evenly spaced across the arc

Two reference frames:
  - "ego" (default): rays rotate with the lander. When the lander tilts,
    the ray fan tilts with it. This is how a real altimeter works — bolted
    to the vehicle's body frame.
  - "world": rays always point straight down regardless of lander tilt.
    Removes orientation-from-terrain information. For ablation experiments.

Coordinate system:
  All inputs/outputs use Box2D world coordinates (not the normalized state
  vector coordinates). The RaycastWrapper handles the coord interface.
"""

import math
from typing import Sequence

import numpy as np


def compute_terrain_rays(
    lander_x: float,
    lander_y: float,
    lander_angle: float,
    terrain_segments: Sequence[tuple[float, float, float, float]],
    n_rays: int = 7,
    arc_degrees: float = 120.0,
    max_range: float = 13.33,
    frame: str = "ego",
) -> np.ndarray:
    """Cast terrain-sensing rays from the lander position.

    Args:
        lander_x: Lander x position in world coordinates.
        lander_y: Lander y position in world coordinates.
        lander_angle: Lander body angle in radians (positive = CCW).
        terrain_segments: List of (x1, y1, x2, y2) terrain line segments
            in world coordinates.
        n_rays: Number of rays in the fan (default 7).
        arc_degrees: Total angular spread of the fan in degrees (default 120).
            Rays span +/-(arc_degrees/2) around the center (down) direction.
        max_range: Maximum sensing distance in world units (default 13.33,
            approximately viewport height in world coords).
        frame: "ego" = rays rotate with lander, "world" = fixed downward.

    Returns:
        Float32 array of shape (n_rays,). Each element is a normalized
        distance in [0, 1]: 0 = ground contact, 1 = no hit within range.
    """
    result = np.ones(n_rays, dtype=np.float32)

    if not terrain_segments:
        return result

    # Pre-compute terrain segment arrays for vectorized intersection.
    # Each segment is (x1, y1) -> (x2, y2).
    n_segs = len(terrain_segments)
    seg_p1 = np.array([(s[0], s[1]) for s in terrain_segments])  # (n_segs, 2)
    seg_p2 = np.array([(s[2], s[3]) for s in terrain_segments])  # (n_segs, 2)
    seg_d = seg_p2 - seg_p1  # segment direction vectors

    origin = np.array([lander_x, lander_y])
    arc_rad = math.radians(arc_degrees)

    for i in range(n_rays):
        # Compute ray direction.
        # Offset angle: evenly spaced across the arc, centered on 0.
        if n_rays > 1:
            # Ray 0 = leftmost (negative offset), ray N-1 = rightmost
            offset = arc_rad * (i / (n_rays - 1) - 0.5)
        else:
            offset = 0.0

        if frame == "ego":
            # Ego frame: "down" = (-sin(theta), -cos(theta)) where theta = lander_angle.
            # This is the negative of the lander's local "up" vector (tip).
            # Rotate by offset around this direction.
            base_angle = -math.pi / 2 - lander_angle  # world angle of lander-down
            ray_angle = base_angle + offset
        else:
            # World frame: "down" = -90 deg from +x axis, regardless of lander tilt.
            ray_angle = -math.pi / 2 + offset

        ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])

        # Find closest intersection with any terrain segment.
        # Parametric intersection: origin + t * ray_dir = seg_p1 + s * seg_d
        # Solving: t = cross(seg_p1 - origin, seg_d) / cross(ray_dir, seg_d)
        #          s = cross(seg_p1 - origin, ray_dir) / cross(ray_dir, seg_d)
        # Valid intersection when t > 0 and 0 <= s <= 1.
        min_t = max_range

        for j in range(n_segs):
            # 2D cross product: a x b = a[0]*b[1] - a[1]*b[0]
            denom = ray_dir[0] * seg_d[j, 1] - ray_dir[1] * seg_d[j, 0]
            if abs(denom) < 1e-10:
                # Ray parallel to segment — no intersection.
                continue

            diff = seg_p1[j] - origin
            t = (diff[0] * seg_d[j, 1] - diff[1] * seg_d[j, 0]) / denom
            s = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / denom

            if t >= 0 and 0 <= s <= 1 and t < min_t:
                min_t = t

        # Normalize by max_range: 0 = at origin, 1 = at/beyond max range.
        result[i] = min(min_t / max_range, 1.0)

    return result
