"""Moving-object detector for the rolling orange ball."""

from __future__ import annotations

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError(
        "OpenCV is required for Task 3 dynamic object detection."
    ) from exc

import numpy as np

from tb4_sim.common.types import BoundingBox, DetectionResult
from tb4_sim.common.vision import find_contours


class DynamicObjectDetector:
    """Detect the moving orange ball and estimate TTC."""

    def __init__(self, ball_diameter_m: float = 0.064) -> None:
        self.ball_diameter_m = ball_diameter_m
        self.previous_blurred_gray: np.ndarray | None = None

    def detect(
        self,
        frame: np.ndarray,
        focal_length_px: float | None,
        robot_speed_mps: float,
    ) -> DetectionResult | None:
        search_top = int(0.30 * frame.shape[0])
        roi = frame[search_top:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Keep the orange mask fairly tight so cardboard boxes are not treated
        # as the rolling ball.
        lower_orange = np.array([8, 140, 120], dtype=np.uint8)
        upper_orange = np.array([22, 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        if self.previous_blurred_gray is None:
            motion_mask = np.full_like(color_mask, 255)
        else:
            frame_delta = cv2.absdiff(blurred, self.previous_blurred_gray)
            _, motion_mask = cv2.threshold(frame_delta, 18, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(
                motion_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=2,
            )
        self.previous_blurred_gray = blurred

        combined_mask = cv2.bitwise_and(color_mask, motion_mask)
        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2,
        )

        contours = find_contours(combined_mask)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 120.0:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            return None

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / max(1.0, float(height))
        if circularity < 0.45 or not (0.60 <= aspect_ratio <= 1.65):
            return None

        bbox = BoundingBox(x=int(x), y=int(y + search_top), width=int(width), height=int(height))

        (_, _), radius = cv2.minEnclosingCircle(contour)
        distance_m = None
        if focal_length_px and radius > 2.0:
            distance_m = float(focal_length_px) * self.ball_diameter_m / max(1.0, 2.0 * radius)

        ttc_seconds = None
        if distance_m is not None and robot_speed_mps > 0.03:
            ttc_seconds = distance_m / robot_speed_mps

        # Danger stop is tuned for the real robot demo where the ball is rolled
        # once across the path. In simulation the ball oscillates continuously
        # so we require it to be extremely large (within ~20cm) to avoid false stops.
        in_path = (bbox.y + bbox.height) > int(0.90 * frame.shape[0])
        close_enough = area > 8000.0
        return DetectionResult(
            label="MOVING",
            bbox=bbox,
            score=min(1.0, area / 2500.0),
            distance_m=distance_m,
            ttc_seconds=ttc_seconds,
            metadata={"danger": in_path and close_enough},
        )
