"""Small geometry helpers shared across task modules."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into the provided inclusive range."""
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert a quaternion into a planar yaw angle."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def order_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Return four points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {pts.shape!r}")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered

