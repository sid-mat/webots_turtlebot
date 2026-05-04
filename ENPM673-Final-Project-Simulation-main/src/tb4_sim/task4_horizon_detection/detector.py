"""Horizon-line estimator tied to the arrow ROI."""

from __future__ import annotations

import math

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError(
        "OpenCV is required for Task 4 horizon detection."
    ) from exc

import numpy as np

from tb4_sim.common.types import BoundingBox


class HorizonDetector:
    """Estimate a stable horizon line using a search region anchored to the arrow ROI."""

    def __init__(self) -> None:
        self.last_line: tuple[int, int, int, int] | None = None

    def _default_line(self, frame_height: int, frame_width: int) -> tuple[int, int, int, int]:
        return (0, int(0.40 * frame_height), frame_width - 1, int(0.40 * frame_height))

    def _smooth_line(self, new_line: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self.last_line is None:
            self.last_line = new_line
            return new_line

        smoothed = tuple(
            int(0.65 * old_value + 0.35 * new_value)
            for old_value, new_value in zip(self.last_line, new_line)
        )
        self.last_line = smoothed
        return smoothed

    def _clamp_full_width_line(
        self,
        slope: float,
        intercept: float,
        frame_height: int,
        frame_width: int,
    ) -> tuple[int, int, int, int]:
        left_y = int(np.clip(intercept, 0, frame_height - 1))
        right_y = int(np.clip(slope * (frame_width - 1) + intercept, 0, frame_height - 1))
        return (0, left_y, frame_width - 1, right_y)

    def _fit_gradient_horizon(
        self,
        response: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        frame_height: int,
        frame_width: int,
    ) -> tuple[int, int, int, int] | None:
        roi_height, roi_width = response.shape[:2]
        if roi_height < 12 or roi_width < 20:
            return None

        band_start = int(0.18 * roi_height)
        band_end = max(band_start + 8, int(0.88 * roi_height))
        band = response[band_start:band_end, :]
        if band.size == 0:
            return None

        row_scores = band.mean(axis=1)
        row_positions = np.linspace(0.0, 1.0, len(row_scores), dtype=np.float32)
        # Prefer the broader lower-middle transition band rather than tiny
        # high edges from posters or trim near the top of the frame.
        row_weights = 0.65 + 0.55 * row_positions
        weighted_scores = row_scores * row_weights

        if float(weighted_scores.max(initial=0.0)) <= 3.0:
            return None

        peak_row = int(np.argmax(weighted_scores)) + band_start
        window_half_height = max(5, int(0.08 * roi_height))
        local_y1 = max(0, peak_row - window_half_height)
        local_y2 = min(roi_height, peak_row + window_half_height + 1)
        local_band = response[local_y1:local_y2, :]
        if local_band.size == 0:
            return None

        sample_step = max(3, roi_width // 60)
        points_x: list[float] = []
        points_y: list[float] = []
        point_weights: list[float] = []
        response_floor = float(np.percentile(local_band, 75))

        for x in range(0, roi_width, sample_step):
            column = local_band[:, x]
            strength = float(column.max(initial=0.0))
            if strength < response_floor:
                continue

            local_row = int(np.argmax(column)) + local_y1
            points_x.append(float(roi_x1 + x))
            points_y.append(float(roi_y1 + local_row))
            point_weights.append(strength)

        if len(points_x) < 6:
            horizon_y = roi_y1 + peak_row
            return (0, horizon_y, frame_width - 1, horizon_y)

        slope, intercept = np.polyfit(
            np.asarray(points_x, dtype=np.float32),
            np.asarray(points_y, dtype=np.float32),
            deg=1,
            w=np.asarray(point_weights, dtype=np.float32),
        )

        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None

        slope = float(np.clip(slope, -0.20, 0.20))
        return self._clamp_full_width_line(slope, float(intercept), frame_height, frame_width)

    def detect(
        self,
        frame: np.ndarray,
        arrow_bbox: BoundingBox | None,
    ) -> tuple[int, int, int, int]:
        frame_height, frame_width = frame.shape[:2]

        if arrow_bbox is not None:
            x_padding = int(0.65 * arrow_bbox.width)
            roi_x1 = max(0, arrow_bbox.x - x_padding)
            roi_x2 = min(frame_width, arrow_bbox.x + arrow_bbox.width + x_padding)
            roi_y1 = max(0, int(0.10 * frame_height))
            roi_y2 = max(
                roi_y1 + 25,
                min(frame_height, arrow_bbox.y + int(0.12 * arrow_bbox.height)),
            )
        else:
            roi_x1 = 0
            roi_x2 = frame_width
            roi_y1 = int(0.10 * frame_height)
            roi_y2 = int(0.60 * frame_height)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return self._smooth_line(self._default_line(frame_height, frame_width))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        horizontal_response = cv2.convertScaleAbs(np.abs(grad_y))

        gradient_line = self._fit_gradient_horizon(
            horizontal_response,
            roi_x1,
            roi_y1,
            frame_height,
            frame_width,
        )
        if gradient_line is not None:
            return self._smooth_line(gradient_line)

        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=45,
            minLineLength=max(25, int(0.35 * roi.shape[1])),
            maxLineGap=30,
        )

        best_line = None
        best_score = -math.inf

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 25.0:
                    continue
                slope = abs(dy / dx) if dx != 0 else float("inf")
                if slope > 0.30:
                    continue

                average_y = (y1 + y2) / 2.0
                score = length + 0.35 * average_y
                if score > best_score:
                    best_score = score
                    best_line = (
                        roi_x1 + x1,
                        roi_y1 + y1,
                        roi_x1 + x2,
                        roi_y1 + y2,
                    )

        if best_line is None:
            if self.last_line is not None:
                return self.last_line
            return self._smooth_line(self._default_line(frame_height, frame_width))

        x1, y1, x2, y2 = best_line
        if x2 == x1:
            full_line = (x1, y1, x2, y2)
        else:
            slope = (y2 - y1) / float(x2 - x1)
            intercept = y1 - slope * x1
            full_line = self._clamp_full_width_line(
                slope,
                intercept,
                frame_height,
                frame_width,
            )
        return self._smooth_line(full_line)
