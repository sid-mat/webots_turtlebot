"""Arrow-sign detector used for Task 1 path following."""

from __future__ import annotations

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError(
        "OpenCV is required for Task 1 arrow detection."
    ) from exc

import numpy as np

from tb4_sim.common.types import BoundingBox, DetectionResult
from tb4_sim.common.vision import find_contours, warp_quadrilateral


class ArrowDetector:
    """Detect the white arrow paper on the floor and classify its direction."""

    def __init__(self, template_path: str, warp_size: int = 128) -> None:
        self.template_path = template_path
        self.warp_size = warp_size
        self.templates = self._build_templates()

    def _build_templates(self) -> dict[str, np.ndarray]:
        """Build one binary template per direction from the arrow_paper.png."""
        img = cv2.imread(self.template_path)
        if img is None:
            raise FileNotFoundError(f"Arrow template not found: {self.template_path}")
        img = cv2.resize(img, (self.warp_size, self.warp_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, tmpl = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Template is an UP arrow — rotate to get all 4 directions.
        return {
            "up":    tmpl,
            "right": cv2.rotate(tmpl, cv2.ROTATE_90_CLOCKWISE),
            "down":  cv2.rotate(tmpl, cv2.ROTATE_180),
            "left":  cv2.rotate(tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

    def _validate_arrow(self, roi: np.ndarray, x: int, y: int, bw: int, bh: int) -> bool:
        crop = roi[max(0,y):min(roi.shape[0],y+bh), max(0,x):min(roi.shape[1],x+bw)]
        if crop.size == 0:
            return False
        ch, cw = crop.shape[:2]
        crop_area = ch * cw
        if crop_area > 150000 or (cw / float(ch) if ch > 0 else 99) > 6.0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        white_ratio = np.sum(norm > 140) / float(crop_area)
        if white_ratio < 0.30:
            return False

        _, black = cv2.threshold(norm, 80, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(black, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or len(contours) > 20:
            return False

        total_black = sum(cv2.contourArea(c) for c in contours)
        black_ratio = total_black / float(crop_area)
        arrow_sized = sum(
            1 for c in contours
            if crop_area * 0.005 < cv2.contourArea(c) < crop_area * 0.60
        )
        return arrow_sized >= 1 and 0.02 <= black_ratio <= 0.85

    def _classify_direction(self, roi: np.ndarray, cnt: np.ndarray) -> str | None:
        """Warp the paper to a square then match against all 4 arrow templates."""
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cnt)
            corners = cv2.boxPoints(rect).astype(np.float32)

        warped = warp_quadrilateral(roi, corners, size=self.warp_size)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        best_dir = None
        best_score = float("inf")
        for direction, tmpl in self.templates.items():
            diff = cv2.absdiff(warped_bin, tmpl)
            score = float(np.mean(diff))
            if score < best_score:
                best_score = score
                best_dir = direction

        return best_dir

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        fh = frame.shape[0]
        roi_top = int(fh * 0.55)
        roi = frame[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        contours = find_contours(white_mask)

        best_cnt = None
        best_bottom = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000 or area > roi_h * roi_w * 0.12:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > roi_w * 0.85 or bh < 20:
                continue
            if bw / float(bh) > 5.0:
                continue
            bottom = by + bh
            if bottom > best_bottom:
                best_bottom = bottom
                best_cnt = cnt

        if best_cnt is None:
            return None

        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        if not self._validate_arrow(roi, bx, by, bw, bh):
            return None

        direction = self._classify_direction(roi, best_cnt)
        if direction is None:
            return None

        y_full = by + roi_top
        bbox = BoundingBox(x=bx, y=y_full, width=bw, height=bh)
        quad_points = [
            (bx, y_full), (bx + bw, y_full),
            (bx + bw, y_full + bh), (bx, y_full + bh),
        ]
        return DetectionResult(
            label="ARROW",
            bbox=bbox,
            score=1.0,
            quad=quad_points,
            direction=direction,
        )

    def detect_candidates(self, frame: np.ndarray) -> list[DetectionResult]:
        result = self.detect(frame)
        return [result] if result is not None else []
