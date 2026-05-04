#!/usr/bin/env python3
"""
ENPM673 Final Project — Task 2: UMD Logo Detection & Stop
=========================================================

Behavior:
    The robot navigates with the camera as its primary vision sensor.
    Whenever the UMD Terrapin logo appears in the camera feed, the
    robot must:
        1. Stop completely.
        2. Display a labeled red bounding box on screen.
        3. Hold for 3 seconds.
        4. Resume.

Approach (classical feature matching):
    - At startup, the reference UMD logo image is loaded and ORB
      keypoints + binary descriptors are extracted from it.
    - For each incoming camera frame:
        * Extract ORB features.
        * Brute-force match against the template (Hamming distance).
        * Apply Lowe's ratio test (0.75) to keep only confidently
          distinctive matches.
        * If >= MIN_MATCH_COUNT good matches survive, fit a homography
          with RANSAC.
        * Require >= MIN_INLIER_COUNT RANSAC inliers and a geometrically
          sane projected quadrilateral. Both checks are essential to
          eliminate false positives from cluttered scenes.
    - On confirmed detection: the homography projects the template's
      four corners into the live frame; an axis-aligned red bounding
      rectangle is drawn (as required by the spec) along with the
      polygonal outline and a "UMD LOGO" label.
    - A simple state machine (CRUISING -> STOPPED -> COOLDOWN) enforces
      a clean 3-second hold and prevents flicker / re-triggering on the
      same sign.

Why ORB + homography (a "classical" approach):
    - The project spec explicitly allows "a fine-tuned ML model or a
      classical feature-matching approach."
    - ORB is rotation- and scale-invariant, robust to mild lighting
      changes, and runs in real time on the TurtleBot's CPU.
    - No training data, no GPU, no risk of dataset/domain mismatch
      between simulation and the actual demo course.

ROS interface:
    Subscribes : /camera/image_raw/image_color  (sensor_msgs/Image)
    Publishes  : /cmd_vel                       (geometry_msgs/Twist)

Parameters:
    template_path      Path to the reference UMD logo PNG (required).
    image_topic        Camera topic (default /camera/image_raw/image_color).
    cmd_vel_topic      Velocity topic (default /cmd_vel).
    show_window        Show the OpenCV visualization (default True).
    autonomous_drive   If True, this node also drives the robot forward at
                       a slow cruise. Set to False when integrating with
                       another driver (e.g. Task 1 path follower) -- the
                       node will still publish zero-velocity during a
                       UMD stop, so its braking takes priority.

Usage:
    # Standalone (this node drives):
    ros2 run tb4_sim umd_logo_stop --ros-args \\
        -p template_path:=/path/to/logo.png

    # Integrated (Task 1 drives; this node only brakes on logo):
    ros2 run tb4_sim umd_logo_stop --ros-args \\
        -p template_path:=/path/to/logo.png \\
        -p autonomous_drive:=false
"""

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image


# --- Detection tunables ---------------------------------------------------
MIN_MATCH_COUNT     = 18      # min good ORB matches to attempt homography
MIN_INLIER_COUNT    = 12      # min RANSAC inliers required to confirm
LOWE_RATIO          = 0.75    # Lowe's ratio test threshold
ORB_FEATURES        = 1500    # nfeatures cap for the live frame

# --- Behavior tunables ----------------------------------------------------
STOP_HOLD_SECONDS   = 3.0     # required stop duration after detection
COOLDOWN_SECONDS    = 8.0     # ignore further detections for this long
                              # after a stop (prevents same sign re-triggering)
CRUISE_LINEAR_VEL   = 0.12    # m/s while cruising (only used if
                              # autonomous_drive=True)
# --------------------------------------------------------------------------


class UmdLogoStop(Node):
    """Detects the UMD logo with ORB + RANSAC homography and stops the
    robot for a fixed hold duration on each detection."""

    def __init__(self):
        super().__init__("umd_logo_stop")

        # ----- ROS parameters ---------------------------------------------
        default_template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "textures", "logo.png",
        )
        self.declare_parameter("template_path", default_template)
        self.declare_parameter("image_topic", "/camera/image_raw/image_color")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("show_window", True)
        self.declare_parameter("autonomous_drive", True)

        template_path   = self.get_parameter("template_path").value
        image_topic     = self.get_parameter("image_topic").value
        cmd_vel_topic   = self.get_parameter("cmd_vel_topic").value
        self.show_window      = bool(self.get_parameter("show_window").value)
        self.autonomous_drive = bool(self.get_parameter("autonomous_drive").value)

        # ----- Load and prepare the template ------------------------------
        if not os.path.isfile(template_path):
            raise FileNotFoundError(
                f"UMD template image not found: {template_path}")
        tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if tpl is None:
            raise RuntimeError(
                f"OpenCV could not decode {template_path}")

        # Upscale tiny templates so ORB has enough scale-space to work with.
        if max(tpl.shape[:2]) < 240:
            scale = 240.0 / max(tpl.shape[:2])
            tpl = cv2.resize(tpl, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        self.tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        self.tpl_h, self.tpl_w = self.tpl_gray.shape

        # ----- ORB detectors and matcher ----------------------------------
        # Two ORB instances so we can tune the template extraction (more
        # features) separately from the per-frame extraction (capped for
        # real-time performance).
        self.orb_tpl = cv2.ORB_create(
            nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.orb_frame = cv2.ORB_create(
            nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8)

        self.kp_tpl, self.des_tpl = self.orb_tpl.detectAndCompute(
            self.tpl_gray, None)
        if self.des_tpl is None or len(self.kp_tpl) < MIN_MATCH_COUNT:
            self.get_logger().warn(
                "Few features extracted from template - "
                "detection may be unreliable.")

        # Hamming distance is the natural metric for ORB's binary descriptors.
        # crossCheck=False because we're going to do Lowe's ratio test, which
        # requires k-nearest matches (k=2).
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ----- ROS interfaces ---------------------------------------------
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, 10)
        self.pub_vel = self.create_publisher(Twist, cmd_vel_topic, 10)

        # ----- State machine ----------------------------------------------
        # CRUISING : driving (or letting another node drive); detector active
        # STOPPED  : logo detected, holding still
        # After STOPPED elapses, transitions back to CRUISING but with a
        # cooldown deadline that suppresses detection for COOLDOWN_SECONDS.
        self.state = "CRUISING"
        self.stop_started_at = 0.0
        self.cooldown_until  = 0.0

        # 10 Hz drive loop -- only meaningful in autonomous mode.
        self.drive_timer = self.create_timer(0.1, self.drive_tick)

        self.get_logger().info(
            f"UMD logo detector ready. "
            f"Template: '{template_path}' "
            f"({self.tpl_w}x{self.tpl_h}, {len(self.kp_tpl)} keypoints). "
            f"Subscribed to {image_topic}. "
            f"Autonomous drive: {self.autonomous_drive}."
        )

    # ----------------------------------------------------------------------
    # Detection
    # ----------------------------------------------------------------------
    def detect_logo(self, frame_gray):
        """Run ORB matching + RANSAC homography against the template.

        Returns:
            (corners, n_inliers) on success, where corners is a (4, 2)
            float array of pixel coordinates of the projected template
            corners (TL, TR, BR, BL).
            (None, n_inliers_or_0) on failure.
        """
        # 1) Per-frame ORB features
        kp_f, des_f = self.orb_frame.detectAndCompute(frame_gray, None)
        if des_f is None or len(kp_f) < MIN_MATCH_COUNT:
            return None, 0

        # 2) k-NN match (k=2) -> Lowe's ratio test
        try:
            knn = self.bf.knnMatch(self.des_tpl, des_f, k=2)
        except cv2.error:
            return None, 0
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair]
                if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_MATCH_COUNT:
            return None, 0

        # 3) Fit a homography with RANSAC
        src_pts = np.float32(
            [self.kp_tpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None, 0
        inliers = int(mask.sum())
        if inliers < MIN_INLIER_COUNT:
            return None, inliers

        # 4) Project template corners through the homography
        corners_tpl = np.float32([
            [0, 0],
            [self.tpl_w - 1, 0],
            [self.tpl_w - 1, self.tpl_h - 1],
            [0, self.tpl_h - 1],
        ]).reshape(-1, 1, 2)
        corners = cv2.perspectiveTransform(corners_tpl, H).reshape(-1, 2)

        # 5) Sanity-check the projected quadrilateral
        if not self._quad_is_reasonable(corners):
            return None, inliers

        return corners, inliers

    @staticmethod
    def _quad_is_reasonable(corners):
        """Reject degenerate homographies (the usual source of false
        positives): vanishing area, non-convex, absurd aspect ratio,
        or collapsed to a thin sliver."""
        area = cv2.contourArea(corners.astype(np.float32))
        if area < 400:
            return False
        hull = cv2.convexHull(corners.astype(np.float32))
        if len(hull) != 4:
            return False
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        if w < 15 or h < 15:
            return False
        if max(w, h) / max(min(w, h), 1) > 5.0:
            return False
        return True

    # ----------------------------------------------------------------------
    # Image callback
    # ----------------------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failure: {e}")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        vis = frame.copy()

        # Run detection only while cruising and outside cooldown -- saves
        # CPU during the 3-second hold and the post-stop debounce window.
        corners, inliers = (None, 0)
        if self.state == "CRUISING" and now >= self.cooldown_until:
            corners, inliers = self.detect_logo(gray)

        if corners is not None:
            self._draw_detection(vis, corners, inliers)
            if self.state == "CRUISING":
                self.get_logger().info(
                    f"UMD logo detected ({inliers} inliers) - STOPPING "
                    f"for {STOP_HOLD_SECONDS:.0f}s."
                )
                self.state = "STOPPED"
                self.stop_started_at = now
                self._publish_zero_vel()  # immediate brake

        if self.state == "STOPPED":
            # Force zero velocity every frame, even if some other node is
            # publishing -- the most recent publish wins on /cmd_vel.
            self._publish_zero_vel()
            elapsed = now - self.stop_started_at
            cv2.putText(
                vis,
                f"STOPPED  {max(0.0, STOP_HOLD_SECONDS - elapsed):.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2, cv2.LINE_AA)
            if elapsed >= STOP_HOLD_SECONDS:
                self.get_logger().info("Resuming.")
                self.state = "CRUISING"
                self.cooldown_until = now + COOLDOWN_SECONDS

        # Status HUD
        cv2.putText(vis, f"state: {self.state}",
                    (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)

        if self.show_window:
            cv2.imshow("Task 2 - UMD Logo Detection", vis)
            cv2.waitKey(1)

    # ----------------------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------------------
    @staticmethod
    def _draw_detection(vis, corners, inliers):
        """Overlay the detection: polygon outline + axis-aligned bounding
        rectangle (the 'red bounding box' the spec requires) + label."""
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        # Quadrilateral outline (anti-aliased red)
        cv2.polylines(vis, [pts], isClosed=True,
                      color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
        # Axis-aligned bounding rectangle
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Label with filled background for readability
        label = f"UMD LOGO  ({inliers} inl.)"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y - 8, th + 4)
        cv2.rectangle(vis, (x, ly - th - 4), (x + tw + 6, ly + 2),
                      (0, 0, 255), -1)
        cv2.putText(vis, label, (x + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ----------------------------------------------------------------------
    # Drive loop
    # ----------------------------------------------------------------------
    def drive_tick(self):
        """Periodic publisher for the cruise / stop velocity."""
        if not self.autonomous_drive:
            # In passive mode, only publish zero-velocity during a STOP
            # event -- so the node still enforces the 3-second hold even
            # when another node (e.g. Task 1 path follower) is the
            # primary driver.
            if self.state == "STOPPED":
                self._publish_zero_vel()
            return

        if self.state == "CRUISING":
            tw = Twist()
            tw.linear.x = CRUISE_LINEAR_VEL
            self.pub_vel.publish(tw)
        elif self.state == "STOPPED":
            self._publish_zero_vel()

    def _publish_zero_vel(self):
        self.pub_vel.publish(Twist())  # default Twist is all zeros


def main():
    rclpy.init()
    node = UmdLogoStop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero_vel()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
