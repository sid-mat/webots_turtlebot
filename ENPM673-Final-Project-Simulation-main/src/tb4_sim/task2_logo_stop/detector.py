"""Logo detector for the UMD Terrapin sign."""

from __future__ import annotations

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError("OpenCV is required for Task 2 logo detection.") from exc

import numpy as np

from tb4_sim.common.types import BoundingBox, DetectionResult
from tb4_sim.common.vision import find_contours


class LogoDetector:
    """Detect the UMD logo using ORB first, then a color fallback."""

    def __init__(self, template_path: str) -> None:
        self.template_path = template_path
        # A more permissive ORB setup helps with partial or oblique views of
        # the logo sign in both simulation and the real robot run.
        self.orb = cv2.ORB_create(nfeatures=1800, fastThreshold=5)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        template = cv2.imread(self.template_path)
        if template is None:
            raise FileNotFoundError(f"Logo template not found at {self.template_path}")

        self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template_height, self.template_width = self.template_gray.shape[:2]
        self.template_keypoints, self.template_descriptors = self.orb.detectAndCompute(
            self.template_gray,
            None,
        )
        if self.template_descriptors is None or len(self.template_keypoints) < 8:
            raise RuntimeError("The UMD logo template did not produce enough ORB features.")

    def detect(self, frame: np.ndarray) -> DetectionResult | None:
        orb_result = self._detect_orb(frame)
        if orb_result is not None:
            return orb_result

        return self._detect_color_fallback(frame)

    def _detect_orb(self, frame: np.ndarray) -> DetectionResult | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 8:
            return None

        matches = self.matcher.knnMatch(self.template_descriptors, descriptors, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.85 * second.distance:
                good_matches.append(first)

        if len(good_matches) < 6:
            return None

        template_points = np.float32(
            [self.template_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        frame_points = np.float32(
            [keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            template_points,
            frame_points,
            cv2.RANSAC,
            6.0,
        )
        if homography is None or inlier_mask is None:
            return None

        inlier_count = int(inlier_mask.ravel().sum())
        if inlier_count < 4:
            return None

        template_corners = np.float32(
            [
                [0, 0],
                [self.template_width - 1, 0],
                [self.template_width - 1, self.template_height - 1],
                [0, self.template_height - 1],
            ]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(template_corners, homography).reshape(-1, 2)

        frame_height, frame_width = frame.shape[:2]
        projected[:, 0] = np.clip(projected[:, 0], 0, frame_width - 1)
        projected[:, 1] = np.clip(projected[:, 1], 0, frame_height - 1)

        quad = [(int(x), int(y)) for x, y in projected]
        bbox = BoundingBox.from_points(quad)
        if bbox.area < 200:
            return None
        if bbox.area > 0.60 * frame_width * frame_height:
            return None

        return DetectionResult(
            label="UMD LOGO",
            bbox=bbox,
            quad=quad,
            score=float(inlier_count) / max(1.0, float(len(good_matches))),
            metadata={"method": "orb"},
        )

    def _detect_color_fallback(self, frame: np.ndarray) -> DetectionResult | None:
        """Fallback for the Maryland logo when ORB misses partial views."""
        frame_height, frame_width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 80, 60], dtype=np.uint8)
        upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([165, 80, 60], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        lower_yellow = np.array([18, 70, 70], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(red_mask, yellow_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours = find_contours(mask)
        if not contours:
            return None

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 8 or height < 8:
                continue

            aspect_ratio = width / max(1.0, float(height))
            if aspect_ratio < 0.25 or aspect_ratio > 4.5:
                continue

            bbox_area = width * height
            if bbox_area > 0.70 * frame_width * frame_height:
                continue

            if y > 0.80 * frame_height and bbox_area < 1500:
                continue

            candidates.append((bbox_area, x, y, width, height))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        _, x, y, width, height = candidates[0]

        bbox = BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height))
        quad = [
            (int(x), int(y)),
            (int(x + width), int(y)),
            (int(x + width), int(y + height)),
            (int(x), int(y + height)),
        ]

        return DetectionResult(
            label="UMD LOGO",
            bbox=bbox,
            quad=quad,
            score=0.70,
            metadata={"method": "color_fallback"},
        )
