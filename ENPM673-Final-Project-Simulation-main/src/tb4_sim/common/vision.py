"""OpenCV helper utilities shared by task modules."""

from __future__ import annotations

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError(
        "OpenCV is required for the perception pipeline. "
        "Install python3-opencv and ros-humble-cv-bridge."
    ) from exc

import numpy as np

from .geometry import order_points


def find_contours(binary_image: np.ndarray) -> list[np.ndarray]:
    """OpenCV version-safe contour extraction."""
    result = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def warp_quadrilateral(image: np.ndarray, points: np.ndarray, size: int = 240) -> np.ndarray:
    """Perspective-warp a quadrilateral patch into a square view."""
    ordered = order_points(points)
    destination = np.array(
        [
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image, matrix, (size, size))


def threshold_dark_regions(image: np.ndarray) -> np.ndarray:
    """Return a mask where darker regions are white."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def rotate_image(image: np.ndarray, clockwise_quarter_turns: int) -> np.ndarray:
    """Rotate an image by 90-degree increments."""
    turns = clockwise_quarter_turns % 4
    if turns == 0:
        return image.copy()
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

