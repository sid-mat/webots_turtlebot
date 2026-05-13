#!/usr/bin/env python3
"""
ENPM673 Spring 2026 - TurtleBot4 Final Project
Complete Perception & Navigation Pipeline

Tasks:
  1. Arrow Paper Following              (25 pts) - green bbox, offset centering
  2. UMD Logo Detection & Stop         (25 pts) - red bbox, 3s stop
  3. Dynamic Ball Detection + TTC      (25 pts) - yellow "MOVING" bbox
  4. Horizon Detection (RANSAC)        (25 pts) - cyan line overlay

Camera topic (sim):  /camera/image_raw
Camera topic (real): /oakd/rgb/preview/image_raw

Standalone usage (no ROS2):
  python3 perception_node.py --test [0|video.mp4]    # all modules
  python3 perception_node.py --stress arrow   [0|file]
  python3 perception_node.py --stress horizon [0|file]
  python3 perception_node.py --stress ball    [0|file]
  python3 perception_node.py --stress umd     [0|file]
"""

# ---- IMPORTS -----------------------------------------------------------------
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
from collections import deque
import time
import math
import os
import sys
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

TB_NUMBER = "tb4_5"
# ---- CONFIGURATION -----------------------------------------------------------

class Config:
    # Topics
    CAMERA_TOPIC  = f"/{TB_NUMBER}/oakd/rgb/preview/image_raw/compressed"
    CMD_VEL_TOPIC = f"/{TB_NUMBER}/cmd_vel"
    ODOM_TOPIC    = f"/{TB_NUMBER}/odom"
    VIZ_TOPIC     = f"/{TB_NUMBER}/enpm673/perception_viz"

    # Image
    IMG_W = 640
    IMG_H = 480

    # Task 1: YOLO arrow detector
    ARROW_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "best.pt"
    )
    ARROW_CONF = 0.20

    # Task 1: paper following
    FORWARD_SPEED  = 0.10   # m/s
    CENTERING_KP   = 0.45    # angular = -KP * offset
    MAX_ANG_SPEED  = 0.30   # rad/s clamp
    MAX_LIN_SPEED  = 0.30   # m/s clamp
    NO_ARROW_LIMIT = 45     # frames without paper -> done
    END_TIMEOUT_S  = 30.0   # safety timeout (seconds)

    # Performance: skip heavy detectors on most frames (run every N)
    UMD_SKIP_N     = 1      # run UMD  every 4th frame (ORB is expensive)
    HORIZ_SKIP_N   = 1      # run horizon every 2nd frame

    # Task 2: UMD
    UMD_TEMPLATE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "umd_logo.png")
    UMD_STOP_DURATION = 3.0
    UMD_MIN_MATCHES   = 10
    UMD_LOWE_RATIO    = 0.75
    UMD_INLIER_THRESH = 5.0
    UMD_MIN_INLIERS   = 6
    UMD_COOLDOWN_S    = 8.0

    # Task 3: ball
    MOG2_HISTORY       = 40
    MOG2_VAR_THRESHOLD = 55
    BALL_MIN_AREA      = 400
    BALL_MAX_AREA      = 50000
    BALL_MIN_CIRC      = 0.35
    BALL_CLEAR_FRAMES  = 15
    BALL_FOCAL_PX      = 380
    BALL_DIAM_M        = 0.20
    BALL_PATH_FRAC_L   = 0.20
    BALL_PATH_FRAC_R   = 0.80
    # Task 3: YOLO ball detector
    BALL_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "best_ball.pt"
    )
    BALL_CONF = 0.80 # only keep detection with conf >25
    BALL_IMGSZ = 320

    # Task 4: horizon RANSAC
    # Task 4 - horizon (tunable)
    HORIZ_EMA     = 0.35   # EMA weight on new estimate (lower = smoother)

    SHOW_DEBUG    = True
    WIN_NAME      = "ENPM673"
    HORIZON_EMA_ALPHA   = 0.45
    HORIZON_SAMPLE_COL  = 12    # scan every N columns
    HORIZON_RANSAC_N    = 80    # iterations
    HORIZON_INLIER_PX   = 15   # px
    HORIZON_MIN_IN      = 6    # min inliers
    HORIZON_MAX_SLOPE   = 1.0  # reject lines steeper than 45 deg
    HORIZON_STALE_RESET = 20   

    # Display
    SHOW_DEBUG = True
    WIN_NAME   = "ENPM673 Perception"


# ---- DATA CLASSES ------------------------------------------------------------

@dataclass
class ArrowResult:
    offset:   float           = 0.0
    bbox:     Optional[Tuple] = None
    detected: bool            = False
    done:     bool            = False

@dataclass
class UMDResult:
    detected: bool            = False
    bbox:     Optional[Tuple] = None

@dataclass
class BallResult:
    detected: bool            = False
    bbox:     Optional[Tuple] = None
    ttc:      Optional[float] = None
    in_path:  bool            = False

@dataclass
class HorizonResult:
    pt1:      Optional[Tuple] = None
    pt2:      Optional[Tuple] = None
    detected: bool            = False


# ---- TASK 1: ARROW / PAPER DETECTOR ------------------------------------------

class ArrowDetector:
    """
    YOLO-based arrow detector.

    Uses trained best.pt to detect black arrow signs.
    Returns:
      ArrowResult.offset   -> center offset in [-1, 1]
      ArrowResult.bbox     -> (x, y, w, h)
      ArrowResult.detected -> True/False
      ArrowResult.done     -> True after too many missing frames

    Draws green bounding rectangle as required by Task 1.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.offset_buffer = deque(maxlen=5)
        self.no_arrow_count = 0

        if not os.path.exists(cfg.ARROW_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO arrow model not found: {cfg.ARROW_MODEL_PATH}\n"
                f"Put best.pt in the same folder as this script."
            )

        self.model = YOLO(cfg.ARROW_MODEL_PATH)

    def detect(self, frame: np.ndarray) -> Tuple[ArrowResult, np.ndarray]:
        result = ArrowResult()
        display = frame.copy()

        h, w = frame.shape[:2]


        # Sharpen blurry live frames before feeding to YOLO
        # Sharpen blurry live frames before feeding to YOLO
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        frame_input = cv2.filter2D(frame, -1, kernel)

        # ── Debug: show before / after edge boost ──────────────────────────
        if self.cfg.SHOW_DEBUG:
            label_before = frame.copy()
            label_after  = frame_input.copy()
            cv2.putText(label_before, "BEFORE (raw)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(label_after,  "AFTER  (edge boost)",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            side_by_side = np.hstack([label_before, label_after])
            cv2.imshow("Arrow: Before / After Sharpen", side_by_side)
        # ───────────────────────────────────────────────────────────────────


        yolo_results = self.model(
            frame_input,
            conf=self.cfg.ARROW_CONF,
            imgsz=320, #320 if lag #416
            verbose=False
        )

        detections = []

        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    continue

                area = bw * bh
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                offset = (cx - (w / 2.0)) / (w / 2.0)

                # Score prefers the closest arrow:
                # large box + lower in image
                score = area * 0.75 + y2 * 20.0

                detections.append({
                    "bbox": (x1, y1, bw, bh),
                    "xyxy": (x1, y1, x2, y2),
                    "conf": conf,
                    "area": area,
                    "center": (cx, cy),
                    "offset": offset,
                    "bottom": y2,
                    "score": score
                })

        if len(detections) == 0:
            self.no_arrow_count += 1
            self.offset_buffer.clear()

            result.offset = 0.0
            result.bbox = None
            result.detected = False
            result.done = self.no_arrow_count >= self.cfg.NO_ARROW_LIMIT

            cv2.putText(
                display,
                "NO ARROW",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            return result, display

        self.no_arrow_count = 0

        # Pick closest / most useful arrow
        detections.sort(key=lambda d: d["score"], reverse=True)
        best = detections[0]

        x, y, bw, bh = best["bbox"]
        x1, y1, x2, y2 = best["xyxy"]

        raw_offset = best["offset"]
        self.offset_buffer.append(raw_offset)
        smooth_offset = float(np.mean(self.offset_buffer))

        result.offset = smooth_offset
        result.bbox = (x, y, bw, bh)
        result.detected = True
        result.done = False

        # Draw all detected arrows in green
        for d in detections:
            dx1, dy1, dx2, dy2 = d["xyxy"]
            conf = d["conf"]

            cv2.rectangle(
                display,
                (dx1, dy1),
                (dx2, dy2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"arrow {conf:.2f}",
                (dx1, max(25, dy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # Highlight selected arrow thicker
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            6
        )

        cx = int((x1 + x2) / 2)

        cv2.line(display, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(display, (cx, y1), (cx, y2), (0, 255, 255), 2)

        cv2.putText(
            display,
            f"SELECTED offset={smooth_offset:+.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        return result, display
 
# ── standalone test ────────────────────────────────────────────────────────────
 
# if __name__ == "__main__":
#     import sys, time
 
#     cfg = Config()
#     det = ArrowDetector(cfg)
 
#     src = sys.argv[1] if len(sys.argv) > 1 else "0"
#     cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
#     if not cap.isOpened():
#         print(f"Cannot open: {src}")
#         raise SystemExit(1)
 
#     cv2.namedWindow("ArrowDetector Test", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("ArrowDetector Test", 960, 540)
 
#     frame_n    = 0
#     det_count  = 0
#     t_start    = time.time()
 
#     print(f"{'Frame':>6}  {'Stage':>8}  {'Det':>5}  {'Offset':>8}  {'BBox'}")
#     print("-" * 65)
 
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue
 
#         frame   = cv2.resize(frame, (cfg.IMG_W, cfg.IMG_H))
#         frame_n += 1
 
#         result, viz = det.detect(frame)
 
#         if result.detected:
#             det_count += 1
 
#         fps  = frame_n / max(time.time() - t_start, 1e-3)
#         rate = det_count / max(frame_n, 1) * 100
 
#         cv2.putText(viz,
#                     f"Frame:{frame_n}  FPS:{fps:.1f}  "
#                     f"Det:{result.detected}  Off:{result.offset:+.2f}  "
#                     f"Rate:{rate:.0f}%",
#                     (5, viz.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
 
#         print(f"{frame_n:>6}  {'DETECT':>8}  {str(result.detected):>5}  "
#               f"{result.offset:>+8.3f}  {str(result.bbox)}")
 
#         cv2.imshow("ArrowDetector Test", viz)
#         k = cv2.waitKey(30) & 0xFF
#         if k == ord('q'):
#             break
 
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"\nTotal: {frame_n} frames, {det_count} detections "
#           f"({det_count/max(frame_n,1)*100:.1f}%)")


# ---- TASK 2: UMD LOGO DETECTOR -----------------------------------------------

class UMDDetector:
    """ORB feature matching + homography; red/gold color fallback."""

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.orb       = cv2.ORB_create(nfeatures=700, scaleFactor=1.2, nlevels=8)
        self.bf        = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._kp       = None
        self._des      = None
        self._th       = 1
        self._tw       = 1
        self._cooldown = 0.0
        self._load(cfg.UMD_TEMPLATE_PATH)

    def _load(self, path):
        if not os.path.exists(path):
            print(f"[UMD] Template not found: {path}  (color fallback only)")
            return
        tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            return
        self._th, self._tw = tmpl.shape[:2]
        self._kp, self._des = self.orb.detectAndCompute(tmpl, None)
        print(f"[UMD] Template: {len(self._kp) if self._kp else 0} kp  ({self._tw}x{self._th})")

    def detect(self, frame: np.ndarray) -> Tuple[UMDResult, np.ndarray]:
        result = UMDResult()
        viz    = frame.copy()
        if time.time() < self._cooldown:
            return result, viz

        if self._des is not None and len(self._des) >= 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)
            if des is not None and len(des) >= 4:
                # explicit loop 
                raw_matches = self.bf.knnMatch(self._des, des, k=2)
                good = []
                for pair in raw_matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < self.cfg.UMD_LOWE_RATIO * n.distance:
                            good.append(m)

                if len(good) >= self.cfg.UMD_MIN_MATCHES:
                    sp = np.float32([self._kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dp = np.float32([kp[m.trainIdx].pt       for m in good]).reshape(-1, 1, 2)
                    H, msk = cv2.findHomography(sp, dp, cv2.RANSAC, self.cfg.UMD_INLIER_THRESH)
                    inliers = int(msk.ravel().sum()) if msk is not None else 0
                    if H is not None and inliers >= self.cfg.UMD_MIN_INLIERS:
                        corners = np.float32(
                            [[0, 0], [self._tw, 0], [self._tw, self._th], [0, self._th]]
                        ).reshape(-1, 1, 2)
                        proj    = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
                        fh, fw  = frame.shape[:2]
                        x1 = int(np.clip(proj[:, 0].min(), 0, fw-1))
                        y1 = int(np.clip(proj[:, 1].min(), 0, fh-1))
                        x2 = int(np.clip(proj[:, 0].max(), 0, fw-1))
                        y2 = int(np.clip(proj[:, 1].max(), 0, fh-1))
                        if x2 > x1 + 20 and y2 > y1 + 20:
                            result.detected = True
                            result.bbox     = (x1, y1, x2-x1, y2-y1)

        if not result.detected:
            result = self._color(frame, result)

        if result.detected:
            self._cooldown = time.time() + self.cfg.UMD_COOLDOWN_S
            x, y, w, h = result.bbox
            cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 0, 255), 3)
            lbl = "UMD TERRAPIN"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(viz, (x, y-lh-12), (x+lw+6, y), (0, 0, 255), -1)
            cv2.putText(viz, lbl, (x+3, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return result, viz

    def _color(self, frame, result):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1   = cv2.inRange(hsv, np.array([0,  120,  80]), np.array([12,  255, 255]))
        r2   = cv2.inRange(hsv, np.array([165, 120, 80]), np.array([180, 255, 255]))
        gold = cv2.inRange(hsv, np.array([18,  100, 150]), np.array([38, 255, 255]))
        k    = np.ones((30, 30), np.uint8)
        both = cv2.bitwise_and(
            cv2.dilate(cv2.bitwise_or(r1, r2), k),
            cv2.dilate(gold, k)
        )
        cnts, _ = cv2.findContours(both, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            lg = max(cnts, key=cv2.contourArea)
            # NOTE-16: raised threshold from 1500 to 2500 for robustness
            if cv2.contourArea(lg) > 2500:
                result.detected = True
                result.bbox     = cv2.boundingRect(lg)
        return result


# # ---- TASK 3: BALL DETECTOR ---------------------------------------------------
# class BallDetector:
#     """YOLO ball detection + simple TTC."""

#     def __init__(self, cfg: Config):
#         self.cfg = cfg

#         if not os.path.exists(cfg.BALL_MODEL_PATH):
#             raise FileNotFoundError(
#                 f"YOLO ball model not found: {cfg.BALL_MODEL_PATH}\n"
#                 f"Put ball_best.pt in the same folder as this script."
#             )

#         self.model = YOLO(cfg.BALL_MODEL_PATH)

#         self._prev_h = None
#         self._prev_t = None
#         self._tbuf = deque(maxlen=6)
#         self._no_det = 0

#     def detect(self, frame: np.ndarray, robot_speed: float = 0.12) -> Tuple[BallResult, np.ndarray]:
#         result = BallResult()
#         viz = frame.copy()

#         fh, fw = frame.shape[:2]

#         yolo_results = self.model(
#             frame,
#             conf=self.cfg.BALL_CONF,
#             imgsz=self.cfg.BALL_IMGSZ,
#             verbose=False
#         )

#         detections = []

#         for r in yolo_results:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 conf = float(box.conf[0])

#                 x1 = max(0, min(x1, fw - 1))
#                 y1 = max(0, min(y1, fh - 1))
#                 x2 = max(0, min(x2, fw - 1))
#                 y2 = max(0, min(y2, fh - 1))

#                 w = x2 - x1
#                 h = y2 - y1

#                 if w <= 0 or h <= 0:
#                     continue

#                 area = w * h
#                 cx = x1 + w / 2.0
#                 cy = y1 + h / 2.0

#                 # Prefer larger / closer ball
#                 score = area * conf

#                 detections.append({
#                     "bbox": (x1, y1, w, h),
#                     "xyxy": (x1, y1, x2, y2),
#                     "conf": conf,
#                     "area": area,
#                     "center": (cx, cy),
#                     "score": score
#                 })

#         if len(detections) == 0:
#             self._no_det += 1
#             self._prev_h = None
#             self._prev_t = None
#             self._tbuf.clear()
#             return result, viz

#         detections.sort(key=lambda d: d["score"], reverse=True)
#         best = detections[0]

#         x, y, w, h = best["bbox"]
#         x1, y1, x2, y2 = best["xyxy"]
#         cx, cy = best["center"]

#         result.detected = True
#         result.bbox = (x, y, w, h)
#         self._no_det = 0

#         # Ball is in robot path if center is in middle part of image
#         result.in_path = (
#             self.cfg.BALL_PATH_FRAC_L * fw < cx < self.cfg.BALL_PATH_FRAC_R * fw
#         )

#         # TTC estimate based on bbox height growth
#         now = time.time()
#         cur_h = float(h)

#         if self._prev_h is not None and self._prev_t is not None:
#             dt = now - self._prev_t
#             dh = cur_h - self._prev_h

#             if 0.0 < dt < 1.5 and dh > 0.5:
#                 ttc = cur_h * dt / dh
#                 self._tbuf.append(ttc)
#                 result.ttc = float(np.median(self._tbuf))

#             elif robot_speed > 0.02 and max(w, h) > 0:
#                 distance = (
#                     self.cfg.BALL_FOCAL_PX
#                     * self.cfg.BALL_DIAM_M
#                     / max(w, h)
#                 )
#                 result.ttc = distance / robot_speed

#         self._prev_h = cur_h
#         self._prev_t = now

#         # Required yellow bounding box for moving ball
#         cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 3)

#         label = f"BALL {best['conf']:.2f}"
#         cv2.putText(
#             viz,
#             label,
#             (x1, max(25, y1 - 30)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.75,
#             (0, 255, 255),
#             2
#         )

#         cv2.putText(
#             viz,
#             "MOVING",
#             (x1, max(45, y1 - 8)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.75,
#             (0, 255, 255),
#             2
#         )

#         if result.ttc is not None:
#             cv2.putText(
#                 viz,
#                 f"TTC:{result.ttc:.1f}s",
#                 (x1, y2 + 22),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (0, 220, 255),
#                 2
#             )

#         if result.in_path:
#             cv2.putText(
#                 viz,
#                 "IN PATH",
#                 (x1, y2 + 45),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (0, 255, 255),
#                 2
#             )

#         return result, viz

#     def is_cleared(self) -> bool:
#         return self._no_det >= self.cfg.BALL_CLEAR_FRAMES

#     def reset_bg(self):
#         self._prev_h = None
#         self._prev_t = None
#         self._tbuf.clear()
#         self._no_det = 0

# ---- TASK 3: BALL DETECTOR ---------------------------------------------------

class BallDetector:
    """MOG2 + multi-color HSV + circularity + TTC."""
   
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.bg      = cv2.createBackgroundSubtractorMOG2(
            history=cfg.MOG2_HISTORY, varThreshold=cfg.MOG2_VAR_THRESHOLD,
            detectShadows=False)
        self._ph     = None
        self._pt     = None
        self._tbuf   = deque(maxlen=6)
        self._no_det = 0
        self._colors = [
            (np.array([0,   100,  80]), np.array([15,  255, 255])),
            (np.array([165, 100,  80]), np.array([180, 255, 255])),
            (np.array([10,  120,  80]), np.array([28,  255, 255])),
            (np.array([22,  100, 100]), np.array([40,  255, 255])),
            (np.array([40,   80,  60]), np.array([80,  255, 255])),
            (np.array([90,   80,  60]), np.array([130, 255, 255])),
            (np.array([130,  80,  60]), np.array([160, 255, 255])),
        ]

    def detect(self, frame: np.ndarray, robot_speed: float = 0.12) -> Tuple[BallResult, np.ndarray]:
        result = BallResult()
        viz    = frame.copy()
        fh, fw = frame.shape[:2]
        ke     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  ke, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ke, iterations=3)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cm  = np.zeros((fh, fw), np.uint8)
        for lo, hi in self._colors:
            cm = cv2.bitwise_or(cm, cv2.inRange(hsv, lo, hi))
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, ke, iterations=2)

        fused = cv2.bitwise_and(fg, cm)

        cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bs, bb  = 0, None
        for cnt in cnts:
            a = cv2.contourArea(cnt)
            if not (self.cfg.BALL_MIN_AREA < a < self.cfg.BALL_MAX_AREA):
                continue
            p = cv2.arcLength(cnt, True)
            if p < 1:
                continue
            c = 4 * math.pi * a / (p * p)
            if c < self.cfg.BALL_MIN_CIRC:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.4 < w / max(h, 1) < 2.2):
                continue
            s = a * c
            if s > bs:
                bs = s
                bb = (cnt, (x, y, w, h), c)

        if bb is not None:
            cnt, (x, y, w, h), circ = bb
            result.detected = True
            result.bbox     = (x, y, w, h)
            self._no_det    = 0
            result.in_path  = (self.cfg.BALL_PATH_FRAC_L * fw
                                < x + w / 2
                                < self.cfg.BALL_PATH_FRAC_R * fw)

            now = time.time()
            hp  = float(h)
            if self._ph is not None and self._pt is not None and now - self._pt < 1.5:
                dh = hp - self._ph
                if dh > 0.5:
                    self._tbuf.append(hp * (now - self._pt) / dh)
                    result.ttc = float(np.median(self._tbuf))
                elif robot_speed > 0.02 and max(w, h) > 0:
                    result.ttc = (
                        self.cfg.BALL_FOCAL_PX * self.cfg.BALL_DIAM_M / max(w, h)
                    ) / robot_speed
            self._ph = hp
            self._pt = now

            cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(viz, "MOVING",
                        (x, y-28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            if result.ttc:
                cv2.putText(viz, f"TTC:{result.ttc:.1f}s",
                            (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            if result.in_path:
                cv2.putText(viz, "IN PATH",
                            (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            self._no_det += 1
            self._ph      = None
            self._pt      = None
            self._tbuf.clear()

        return result, viz

    def is_cleared(self) -> bool:
        return self._no_det >= self.cfg.BALL_CLEAR_FRAMES

    def reset_bg(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.MOG2_HISTORY,
            varThreshold=self.cfg.MOG2_VAR_THRESHOLD,
            detectShadows=False)

# ── TASK 4: HORIZON DETECTOR ─────────────────────────────────────────────────
class HorizonDetector:
    """
    Two-stage horizon detection robust to wall decorations, frames, and clutter.

    Stage 1 - Row intensity profile (finds Y):
        Average brightness across every full-width row between y=20% and y=85%.
        Apply a heavy 1-D Gaussian smooth (~51px kernel) so picture frames,
        which occupy only part of a row, get averaged away.  The floor is darker
        than the wall; the floor-wall boundary is where the smoothed row-mean
        changes most steeply.  This is robust because the floor-wall transition
        is GLOBAL (every column) while clutter is LOCAL (few columns).

    Stage 2 - Local Hough for slope (finds tilt):
        Search only ±MARGIN px around the y from Stage 1.  Slope filter is
        strict (≤0.15, ~8.5°) because the true horizon is nearly horizontal for
        a forward-facing robot camera.  Picture frames far from the estimated y
        are simply out of the search window.

    ROI note: assignment says use arrow bbox as ROI.  Arrow is on the floor so
    its y-extent is below the horizon.  We use arrow x-extent to optionally
    narrow the Hough search; y bounds come from Stage 1 only.
    """

    SCAN_Y0   = 0.20   # start row search at 20 % of frame height
    SCAN_Y1   = 0.85   # end row search at 85 %
    SMOOTH_K  = 51     # Gaussian kernel for row-mean smoothing (must be odd)
    MARGIN    = 60     # px above/below y_est to search for Hough lines
    MAX_SLOPE = 0.15   # |slope| filter for Hough lines (~8.5 deg)

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self._sm_m = None
        self._sm_b = None

    def detect(self, frame: np.ndarray,
               arrow_roi: Optional[Tuple] = None) -> Tuple[HorizonResult, np.ndarray]:
        viz    = frame.copy()
        fh, fw = frame.shape[:2]
        result = HorizonResult()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Stage 1: robust Y via row-mean brightness profile ─────────────
        ys0   = int(fh * self.SCAN_Y0)
        ys1   = int(fh * self.SCAN_Y1)
        strip = gray[ys0:ys1, :].astype(np.float32)

        # Mean brightness of each row (full width → local objects average out)
        means = strip.mean(axis=1)

        # Heavy smoothing: reflect-pad so edges don't create fake gradients.
        # A 51-row uniform kernel washes out objects occupying < ~25 rows.
        k = min(self.SMOOTH_K, (len(means) // 2) * 2 - 1)  # odd, fits in array
        if k >= 3:
            pad    = np.pad(means, k // 2, mode='reflect')
            means  = np.convolve(pad, np.ones(k) / k, mode='valid')

        # Max |gradient| row = sharpest brightness transition = horizon
        grad  = np.abs(np.diff(means))
        y_est = ys0 + int(np.argmax(grad))

        # ── Stage 2: Hough lines in ±MARGIN window around y_est ──────────
        # Optional x restriction from arrow bbox (cross-ratio: same y answer
        # in any x strip, but narrowing reduces false positives)
        if arrow_roi:
            ax, ay, aw, ah = arrow_roi
            x1 = max(0, ax - max(80, aw));  x2 = min(fw, ax + aw + max(80, aw))
        else:
            x1, x2 = 0, fw

        ry0  = max(0, y_est - self.MARGIN)
        ry1  = min(fh, y_est + self.MARGIN)
        roi  = frame[ry0:ry1, x1:x2]
        blur = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edg  = cv2.Canny(blur, 50, 150)
        lns  = cv2.HoughLinesP(edg, 1, np.pi/180,
                               threshold=20, minLineLength=40, maxLineGap=30)

        m, b = 0.0, float(y_est)   # default: horizontal at estimated y
        if lns is not None:
            xs, ys_c, ws = [], [], []
            for l in lns:
                lx1, ly1, lx2, ly2 = l[0]
                dx = lx2 - lx1;  dy = ly2 - ly1
                if abs(dx) < 1: continue
                if abs(dy / dx) > self.MAX_SLOPE: continue   # reject steep lines
                lg = math.hypot(dx, dy)
                # Convert coordinates back to full-frame
                xs  += [lx1 + x1,       lx2 + x1      ]
                ys_c+= [ly1 + ry0,      ly2 + ry0      ]
                ws  += [lg / 2,          lg / 2         ]
            if len(xs) >= 2:
                X, Y, W = np.array(xs, float), np.array(ys_c, float), np.array(ws, float)
                sw   = W.sum()
                swx  = (W * X).sum();   swy  = (W * Y).sum()
                swx2 = (W * X * X).sum(); swxy = (W * X * Y).sum()
                denom = sw * swx2 - swx * swx
                if abs(denom) > 1e-6:
                    m = (sw * swxy - swx * swy) / denom
                    b = (swy - m * swx) / sw

        # ── EMA smoothing ─────────────────────────────────────────────────
        a = self.cfg.HORIZ_EMA
        if self._sm_m is None:
            self._sm_m, self._sm_b = m, b
        else:
            # Sanity: reject if y jumped more than 80px in one frame
            new_mid = m * fw/2 + b
            old_mid = self._sm_m * fw/2 + self._sm_b
            if abs(new_mid - old_mid) < 80:
                self._sm_m = a * m + (1 - a) * self._sm_m
                self._sm_b = a * b + (1 - a) * self._sm_b

        # ── Draw ──────────────────────────────────────────────────────────
        if self._sm_m is not None:
            sm, sb = self._sm_m, self._sm_b
            y_l = int(round(sb))
            y_r = int(round(sm * fw + sb))
            pt1 = (0,  int(np.clip(y_l, 0, fh - 1)))
            pt2 = (fw, int(np.clip(y_r, 0, fh - 1)))
            cv2.line(viz, pt1, pt2, (0, 255, 255), 2)
            mid_y = int(round(sm * fw / 2 + sb))
            cv2.putText(viz, f"HORIZON y={mid_y}",
                        (fw // 2 - 65, max(mid_y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            result.pt1, result.pt2, result.detected = pt1, pt2, True

        return result, viz
# ---- NAVIGATOR ---------------------------------------------------------------

class RobotState(Enum):
    MOVING   = auto()
    UMD_STOP = auto()
    BALL_STP = auto()
    DONE     = auto()


class Navigator:
    """
    Continuous offset-based path following. No discrete turn commands.
    Priority: BALL > UMD > ARROW

    ball path detection now also considers the robot's current
    lateral offset so near-edge balls are not missed.
    """

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.state        = RobotState.MOVING
        self._t_state     = time.time()
        self._last_seen_t = time.time()

    def step(self, arrow: ArrowResult, umd: UMDResult,
             ball: BallResult, ball_cleared: bool) -> Tuple[float, float]:
        cfg = self.cfg

        if self.state == RobotState.DONE:
            return 0.0, 0.0

        # Ball: highest priority
    
        if ball.detected:
            self.state = RobotState.BALL_STP
            return 0.0, 0.0
        if self.state == RobotState.BALL_STP:
            if ball_cleared:
                self.state    = RobotState.MOVING
                self._t_state = time.time()
            return 0.0, 0.0

        # UMD: 3-second stop
        if umd.detected and self.state != RobotState.UMD_STOP:
            self.state    = RobotState.UMD_STOP
            self._t_state = time.time()
        if self.state == RobotState.UMD_STOP:
            if time.time() - self._t_state >= cfg.UMD_STOP_DURATION:
                self.state    = RobotState.MOVING
                self._t_state = time.time()
            return 0.0, 0.0

        # Arrow done signal
        if arrow.done:
            self.state = RobotState.DONE
            return 0.0, 0.0

        # Safety timeout
        if arrow.detected:
            self._last_seen_t = time.time()
        if time.time() - self._last_seen_t > cfg.END_TIMEOUT_S:
            self.state = RobotState.DONE
            return 0.0, 0.0

        # Continuous offset centering
        linear  = cfg.FORWARD_SPEED * max(0.3, 1.0 - abs(arrow.offset) * 0.5)
        angular = float(np.clip(-cfg.CENTERING_KP * arrow.offset,
                                -cfg.MAX_ANG_SPEED, cfg.MAX_ANG_SPEED))
        return linear, angular


# ---- MAIN ROS2 NODE ----------------------------------------------------------

class PerceptionNode(Node):

    def __init__(self, cfg: Config):
        super().__init__("enpm673_perception")
        self.cfg = cfg

        self.arrow_det   = ArrowDetector(cfg)
        self.umd_det     = UMDDetector(cfg)
        self.ball_det    = BallDetector(cfg)
        self.horizon_det = HorizonDetector(cfg)
        self.navigator   = Navigator(cfg)
        self.bridge      = CvBridge()

        self.declare_parameter("camera_topic", cfg.CAMERA_TOPIC)
        cam_topic = self.get_parameter("camera_topic").value

        if cfg.CMD_VEL_TOPIC.startswith("/tb4_"):
            self._use_twist_stamped = True
            self._cmd_pub = self.create_publisher(TwistStamped,cfg.CMD_VEL_TOPIC,10)
        else:
            self._use_twist_stamped = False
            self._cmd_pub = self.create_publisher(Twist,cfg.CMD_VEL_TOPIC,10)
        self._viz_pub  = self.create_publisher(Image, cfg.VIZ_TOPIC, 10)
        # camera_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     durability=DurabilityPolicy.VOLATILE,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=10

        # )

        if cam_topic.endswith("/compressed"):
            self._use_compressed = True
            self._img_sub = self.create_subscription(
                CompressedImage,
                cam_topic,
                self._img_cb,
                10
            )
        else:
            self._use_compressed = False
            self._img_sub = self.create_subscription(
                Image,
                cam_topic,
                self._img_cb,
                10
            )
        self._odom_sub = self.create_subscription(Odometry, cfg.ODOM_TOPIC, self._odom_cb, 10)

        self._robot_speed = cfg.FORWARD_SPEED
        self._frame_n     = 0
        self._first_frame = None
        self._umd_skip_c  = 0
        self._hor_skip_c  = 0

        # initialize cached detector results in __init__
        # so the getattr fallbacks in _img_cb always have a valid value
        self._last_ur = UMDResult()
        self._last_uv = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)
        self._last_hr = HorizonResult()
        self._last_hv = np.zeros((cfg.IMG_H, cfg.IMG_W, 3), dtype=np.uint8)

        self._no_img_timer = self.create_timer(5.0, self._check_camera)

        if cfg.SHOW_DEBUG:
            cv2.namedWindow(cfg.WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(cfg.WIN_NAME, 1280, 540)

        self.get_logger().info(f"ENPM673 Perception ready. Camera: {cam_topic}")

    def publish_cmd(self, lin: float, ang: float):

        if self._use_twist_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            msg.twist.linear.x = float(lin)
            msg.twist.angular.z = float(ang)

        else:
            msg = Twist()
            msg.linear.x = float(lin)
            msg.angular.z = float(ang)

        self._cmd_pub.publish(msg)

    def _check_camera(self):
        if self._first_frame is None:
            self.get_logger().warn(
                f"No images on '{self.get_parameter('camera_topic').value}'!\n"                
                f"  ros2 topic list | grep -iE 'image|camera'\n"
                f"  Sim:  /camera/image_raw  (--sim)\n"
                f"  Real: /oakd/rgb/preview/image_raw  (--real)")
        else:
            fps = self._frame_n / max(time.time() - self._first_frame, 1e-3)
            self.get_logger().info(f"Camera OK - {self._frame_n} frames @ {fps:.1f} fps")
        self.destroy_timer(self._no_img_timer)

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._robot_speed = max(math.hypot(vx, vy), 0.01)

    def _img_cb(self, msg: Image):
        t_now = time.time()
        self._frame_n += 1
        if self._first_frame is None:
            self._first_frame = t_now
            self.get_logger().info("First image received - pipeline active")

        # try:
        #     frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # except Exception as e:
        #     self.get_logger().error(f"cv_bridge: {e}")
        #     return
        if self._use_compressed:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Could not decode compressed image")
                return
        else:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                self.get_logger().error(f"cv_bridge: {e}")
                return
            
        frame = cv2.resize(frame, (self.cfg.IMG_W, self.cfg.IMG_H))

        ar, av = self.arrow_det.detect(frame)
        br, bv = self.ball_det.detect(frame, robot_speed=self._robot_speed)

        # UMD: expensive ORB - run every UMD_SKIP_N frames
        self._umd_skip_c = (self._umd_skip_c + 1) % self.cfg.UMD_SKIP_N
        if self._umd_skip_c == 0:
            ur, uv = self.umd_det.detect(frame)
            self._last_ur = ur
            self._last_uv = uv
        else:
            ur, uv = self._last_ur, self._last_uv

        # Horizon: RANSAC - run every HORIZ_SKIP_N frames
        self._hor_skip_c = (self._hor_skip_c + 1) % self.cfg.HORIZ_SKIP_N
        if self._hor_skip_c == 0:
            hr, hv = self.horizon_det.detect(frame, arrow_roi=ar.bbox)
            self._last_hr = hr
            self._last_hv = hv
        else:
            hr, hv = self._last_hr, self._last_hv

        lin, ang = self.navigator.step(ar, ur, br, ball_cleared=self.ball_det.is_cleared())

        lin = float(np.clip(lin, -self.cfg.MAX_LIN_SPEED, self.cfg.MAX_LIN_SPEED))
        ang = float(np.clip(ang, -self.cfg.MAX_ANG_SPEED, self.cfg.MAX_ANG_SPEED))

        self.publish_cmd(lin, ang)

        viz = self._compose(frame, av, uv, bv, hv, ar, ur, br, hr, lin, ang)

        if self.cfg.SHOW_DEBUG:
            cv2.imshow(self.cfg.WIN_NAME, viz)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key == ord('r'):
                self.navigator.state        = RobotState.MOVING
                self.navigator._last_seen_t = time.time()
                self.ball_det.reset_bg()
                self.get_logger().info("Manual reset")

        try:
            vm = self.bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            vm.header = msg.header
            self._viz_pub.publish(vm)
        except Exception:
            pass

    def _compose(self, frame, av, uv, bv, hv, ar, ur, br, hr, lin, ang):
        # build viz incrementally so no detector overwrites another.
        # Previously each annotated frame was diffed against the ORIGINAL frame,
        # meaning the last detector (UMD) could silently erase ball annotations
        # on pixels where both detectors drew.  Now we layer in a fixed order:
        # horizon (background) -> arrow -> ball -> UMD (foreground labels on top).
        viz = frame.copy()
        fh, fw = viz.shape[:2]

        for annotated in (hv, av, bv, uv):
            diff = (annotated.astype(np.int16) - frame.astype(np.int16)) != 0
            viz[diff.any(2)] = annotated[diff.any(2)]

        sc = {"MOVING": (80, 220, 80), "UMD_STOP": (80, 80, 255),
              "BALL_STP": (80, 255, 255), "DONE": (255, 255, 255)}.get(
                  self.navigator.state.name, (200, 200, 200))
        cv2.rectangle(viz, (0, 0), (fw, 32), (20, 20, 20), -1)
        hud = (f"State:{self.navigator.state.name}  Lin:{lin:.2f}  "
               f"Ang:{ang:.2f}  Off:{ar.offset:+.2f}  Fr:{self._frame_n}")
        cv2.putText(viz, hud, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, sc, 2)

        if self.navigator.state == RobotState.UMD_STOP:
            rem = max(0.0, self.cfg.UMD_STOP_DURATION -
                      (time.time() - self.navigator._t_state))
            cv2.putText(viz, f"UMD HOLD: {rem:.1f}s", (fw//2 - 90, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return viz


# ---- STANDALONE ALL-MODULE TEST ----------------------------------------------

def run_standalone(source: str, cfg: Config):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"Cannot open: {source}")
        return
    ad = ArrowDetector(cfg);  ud = UMDDetector(cfg)
    bd = BallDetector(cfg);   hd = HorizonDetector(cfg)
    cv2.namedWindow(cfg.WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WIN_NAME, 1280, 540)
    print("Q=quit  R=reset BG")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (cfg.IMG_W, cfg.IMG_H))
        ar, av = ad.detect(frame)
        ur, uv = ud.detect(frame)
        br, bv = bd.detect(frame)
        hr, hv = hd.detect(frame, arrow_roi=ar.bbox)
        viz = frame.copy()
        for dv in (hv, av, bv, uv):
            diff = (dv.astype(np.int16) - frame.astype(np.int16)) != 0
            viz[diff.any(2)] = dv[diff.any(2)]
        for i, txt in enumerate([
            f"Arrow  off={ar.offset:+.2f}  det={ar.detected}  done={ar.done}",
            f"UMD    det={ur.detected}",
            f"Ball   det={br.detected}  ttc={br.ttc}  path={br.in_path}",
            f"Horiz  det={hr.detected}  pt1={hr.pt1}",
        ]):
            cv2.putText(viz, txt, (5, 22 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)
        cv2.imshow(cfg.WIN_NAME, viz)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            bd.reset_bg()
            print("BG reset")
    cap.release()
    cv2.destroyAllWindows()


# ---- PER-MODULE STRESS TESTER ------------------------------------------------

def run_stress(module: str, source: str, cfg: Config):
    """
    Stress-test a single detector in isolation.

    Source options:
      0, 1, ...       webcam device index
      /path/to.mp4    video file
      ros             read from cfg.CAMERA_TOPIC via ROS2

    Hotkeys: Q=quit  R=reset BG subtractor (ball only)
    """
    use_ros = source.lower() == "ros"

    if use_ros:
        import queue as _queue, threading
        _fq = _queue.Queue(maxsize=2)
        rclpy.init()
        _node = rclpy.create_node("enpm673_stress")
        from cv_bridge import CvBridge as _CVB
        _br = _CVB()
        from sensor_msgs.msg import Image as _Img

        def _cb(msg):
            try:
                f = _br.imgmsg_to_cv2(msg, "bgr8")
                f = cv2.resize(f, (cfg.IMG_W, cfg.IMG_H))
                if not _fq.full():
                    _fq.put_nowait(f)
            except Exception:
                pass

        _node.create_subscription(_Img, cfg.CAMERA_TOPIC, _cb, 10)
        threading.Thread(target=lambda: rclpy.spin(_node), daemon=True).start()
        print(f"[stress] Subscribed to {cfg.CAMERA_TOPIC} - waiting for frames ...")

        def get_frame():
            try:    return _fq.get(timeout=5.0)
            except: return None

        def cleanup():
            try:    _node.destroy_node()
            except: pass
            try:    rclpy.shutdown()
            except: pass
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print(f"Cannot open source: {source}")
            print("Tip: for Webots/real robot use 'ros' as source and add --sim or --real")
            return

        def get_frame():
            ret, f = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = cap.read()
            return cv2.resize(f, (cfg.IMG_W, cfg.IMG_H)) if ret else None

        def cleanup():
            cap.release()

    det       = None
    det_count = 0

    if module == "arrow":
        det = ArrowDetector(cfg)
        print(f"{'Frame':>6}  {'Det':>5}  {'Offset':>8}  {'Done':>5}  {'NoCnt':>6}")
        print("-" * 45)

        def run_one(f, fn):
            nonlocal det_count
            r, v = det.detect(f)
            if r.detected: det_count += 1
            print(f"{fn:>6}  {str(r.detected):>5}  {r.offset:>+8.3f}"
                  f"  {str(r.done):>5}  {det.no_arrow_count:>6}")
            return v

    elif module == "horizon":
        det = HorizonDetector(cfg)
        print(f"{'Frame':>6}  {'Det':>5}  {'y_mid':>6}  {'slope':>9}  {'b':>8}")
        print("-" * 48)

        def run_one(f, fn):
            nonlocal det_count
            r, v = det.detect(f)
            if r.detected:
                det_count += 1
                fw_  = f.shape[1]
                m    = det._sm_m if det._sm_m is not None else 0.0
                b    = det._sm_b if det._sm_b is not None else 0.0
                mid_y = int(round(m * fw_ / 2 + b))
                print(f"{fn:>6}  {str(r.detected):>5}  {mid_y:>6}  {m:>+9.4f}  {b:>8.1f}")
            else:
                print(f"{fn:>6}  {'False':>5}  {'--':>6}  {'--':>9}  {'--':>8}")
            return v

    elif module == "ball":
        det = BallDetector(cfg)
        print(f"{'Frame':>6}  {'Det':>5}  {'InPath':>7}  {'TTC':>7}  {'BBox'}")
        print("-" * 55)

        def run_one(f, fn):
            nonlocal det_count
            r, v = det.detect(f)
            if r.detected: det_count += 1
            ttcs = f"{r.ttc:.2f}" if r.ttc else "  -- "
            bbs  = str(r.bbox) if r.bbox else "--"
            print(f"{fn:>6}  {str(r.detected):>5}  {str(r.in_path):>7}  "
                  f"{ttcs:>7}  {bbs}")
            return v

    elif module == "umd":
        det = UMDDetector(cfg)
        print(f"{'Frame':>6}  {'Detected':>8}  {'BBox'}")
        print("-" * 45)

        def run_one(f, fn):
            nonlocal det_count
            r, v = det.detect(f)
            if r.detected: det_count += 1
            print(f"{fn:>6}  {str(r.detected):>8}  {str(r.bbox)}")
            return v

    else:
        print(f"Unknown module '{module}'. Choose: arrow | horizon | ball | umd")
        cleanup()
        return

    win = f"STRESS: {module.upper()}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    print(f"\n{'='*58}")
    print(f"  STRESS TEST: {module.upper():<12} source={source}")
    print(f"  Q=quit   R=reset BG (ball only)")
    print(f"{'='*58}")

    # single initialization block (was duplicated — second copy
    # accidentally reset counters AFTER run_one was defined, masking the first)
    frame_n   = 0
    t_start   = time.time()
    det_count = 0

    try:
        while True:
            frame = get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            frame_n += 1
            t0  = time.time()
            viz = run_one(frame, frame_n)
            lat = (time.time() - t0) * 1000
            fps  = frame_n / max(time.time() - t_start, 1e-3)
            rate = det_count / max(frame_n, 1) * 100
            cv2.putText(viz,
                        f"Frame:{frame_n}  Lat:{lat:.1f}ms  FPS:{fps:.1f}  "
                        f"DetRate:{rate:.0f}%",
                        (5, viz.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
            cv2.imshow(win, viz)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('r') and module == "ball":
                det.reset_bg()
                print(f"{'':>6}  [BG reset]")
    finally:
        elapsed = time.time() - t_start
        print(f"\n{'='*58}")
        print(f"  Frames    : {frame_n}")
        print(f"  Detections: {det_count}  ({det_count / max(frame_n, 1) * 100:.1f}%)")
        print(f"  Time      : {elapsed:.1f}s  avg {frame_n / max(elapsed, 1e-3):.1f} fps")
        print(f"{'='*58}\n")
        cleanup()
        cv2.destroyAllWindows()


# ---- ENTRY POINT -------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test",   nargs="?", const="0", default=None,
                        help="All-module test: --test [0|video.mp4]")
    parser.add_argument("--stress", nargs="+", default=None,
                        metavar=("MODULE", "SOURCE"),
                        help="Single-module stress: --stress arrow [0|file]")
    parser.add_argument("--sim",    action="store_true")
    parser.add_argument("--real",   action="store_true")
    parser.add_argument("--no-nav", action="store_true")
    known, rest = parser.parse_known_args()

    cfg = Config()

    if known.sim:
        cfg.CAMERA_TOPIC = "/camera/image_raw/image_color"
        cfg.CMD_VEL_TOPIC = "/cmd_vel"
        cfg.ODOM_TOPIC = "/odom"
        cfg.VIZ_TOPIC = "/enpm673/perception_viz"

    elif known.real:
        cfg.CAMERA_TOPIC = f"/{TB_NUMBER}/oakd/rgb/preview/image_raw/compressed"
        cfg.CMD_VEL_TOPIC = f"/{TB_NUMBER}/cmd_vel"
        cfg.ODOM_TOPIC = f"/{TB_NUMBER}/odom"
        cfg.VIZ_TOPIC = f"/{TB_NUMBER}/enpm673/perception_viz"

    if known.test is not None:
        run_standalone(known.test, cfg)
        return

    if known.stress is not None:
        mod = known.stress[0].lower()
        src = known.stress[1] if len(known.stress) > 1 else "0"
        run_stress(mod, src, cfg)
        return

    # ROS2 mode
    rclpy.init(args=rest)
    node = PerceptionNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:    
            node.publish_cmd(0.0, 0.0)
        except: pass
        try:    node.destroy_node()
        except: pass
        try:    rclpy.shutdown()
        except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()