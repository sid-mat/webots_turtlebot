"""Integrated mission controller for the ENPM673 TurtleBot final project."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on system packages
    raise RuntimeError(
        "OpenCV is required for the mission controller."
    ) from exc

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from tb4_sim.common.geometry import clamp, normalize_angle, quaternion_to_yaw
from tb4_sim.common.types import BoundingBox, DetectionResult
from tb4_sim.task1_arrow_following.detector import ArrowDetector
from tb4_sim.task2_logo_stop.detector import LogoDetector
from tb4_sim.task3_dynamic_object.detector import DynamicObjectDetector
from tb4_sim.task4_horizon_detection.detector import HorizonDetector


@dataclass
class PerceptionSnapshot:
    arrow: DetectionResult | None = None
    logo: DetectionResult | None = None
    moving_object: DetectionResult | None = None
    horizon_line: tuple[int, int, int, int] | None = None
    frame_width: int = 0
    frame_height: int = 0
    timestamp_s: float = 0.0


class MissionController(Node):
    """Runs all four project tasks with a simple state-driven controller."""

    def __init__(self) -> None:
        super().__init__("mission_controller")

        self.declare_parameter("image_topic", "/camera/image_raw/image_color")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_vel_stamped", False)
        self.declare_parameter("cmd_vel_frame_id", "")
        self.declare_parameter("annotated_image_topic", "/perception/annotated")
        self.declare_parameter("show_debug_window", False)
        self.declare_parameter("logo_pause_seconds", 3.0)
        self.declare_parameter("turn_speed_rad_s", 0.85)
        self.declare_parameter("cruise_speed_m_s", 0.14)
        self.declare_parameter("search_speed_m_s", 0.12)
        self.declare_parameter("search_turn_speed_rad_s", 0.38)
        self.declare_parameter("centering_gain", 1.55)
        # Keep the default consistent with the Webots launch tuning so Task 1
        # produces a clearly visible turn unless a caller overrides it.
        self.declare_parameter("arrow_turn_angle_rad", 1.15)
        self.declare_parameter("arrow_commit_center_ratio", 0.10)
        self.declare_parameter("arrow_commit_bottom_ratio", 0.92)
        self.declare_parameter("arrow_commit_area_ratio", 0.13)
        self.declare_parameter("arrow_ignore_seconds", 2.4)
        self.declare_parameter("arrow_pass_seconds", 0.15)
        self.declare_parameter("down_arrow_ignore_seconds", 3.0)
        self.declare_parameter("down_arrow_coast_seconds", 1.2)
        self.declare_parameter("arrow_clear_hold_seconds", 0.6)
        self.declare_parameter("post_turn_forward_seconds", 0.45)
        self.declare_parameter("mission_idle_timeout_s", 25.0)
        self.declare_parameter("complete_on_idle_timeout", False)
        self.declare_parameter("down_arrow_is_end_marker", False)
        self.declare_parameter("treat_down_arrow_as_left", True)

        package_share = get_package_share_directory("tb4_sim")
        textures_dir = os.path.join(package_share, "textures")

        self.bridge = CvBridge()
        self.arrow_detector = ArrowDetector(os.path.join(textures_dir, "arrow_paper.png"))
        self.logo_detector = LogoDetector(os.path.join(textures_dir, "logo.png"))
        self.dynamic_detector = DynamicObjectDetector()
        self.horizon_detector = HorizonDetector()

        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.cmd_vel_stamped = bool(self.get_parameter("cmd_vel_stamped").value)
        self.cmd_vel_frame_id = str(self.get_parameter("cmd_vel_frame_id").value)
        self.down_arrow_is_end_marker = bool(
            self.get_parameter("down_arrow_is_end_marker").value
        )
        self.treat_down_arrow_as_left = bool(
            self.get_parameter("treat_down_arrow_as_left").value
        )
        annotated_topic = self.get_parameter("annotated_image_topic").value
        self.show_debug_window = bool(self.get_parameter("show_debug_window").value)

        if self.cmd_vel_stamped:
            self.cmd_publisher = self.create_publisher(TwistStamped, cmd_vel_topic, 10)
        else:
            self.cmd_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.annotated_image_publisher = self.create_publisher(Image, annotated_topic, 10)

        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.snapshot = PerceptionSnapshot()
        self.last_arrow_bbox: BoundingBox | None = None
        self.camera_fx: float | None = None
        self.current_speed_mps = 0.0
        self.current_yaw: float | None = None

        self.executed_arrow_count = 0
        self.logo_completed = False
        self.mission_completed = False
        self.logo_pause_until_s = 0.0
        self.ball_stop_until_s = 0.0
        self.ignore_arrow_until_s = 0.0
        self.forward_drive_until_s = 0.0
        self.turn_target_yaw: float | None = None
        self.turn_end_until_s = 0.0
        self.turn_command_rad_s = 0.0
        self.pending_turn_delta_rad = 0.0
        self.last_arrow_seen_s = 0.0
        self.awaiting_arrow_clear = False
        self.last_committed_arrow_bbox: BoundingBox | None = None
        self.search_turn_sign = 0.0
        self.reacquire_after_logo = False
        self.passed_sign_recovery_until_s = 0.0
        self.idle_recovery_logged = False

        arrow_turn_angle = float(self.get_parameter("arrow_turn_angle_rad").value)
        arrow_pass_seconds = float(self.get_parameter("arrow_pass_seconds").value)
        idle_timeout = float(self.get_parameter("mission_idle_timeout_s").value)
        complete_on_idle_timeout = bool(
            self.get_parameter("complete_on_idle_timeout").value
        )

        self.get_logger().info(
            "Mission controller ready. "
            f"image={image_topic}, camera_info={camera_info_topic}, odom={odom_topic}, "
            f"cmd_vel={cmd_vel_topic}, stamped_cmd_vel={self.cmd_vel_stamped}, "
            f"annotated={annotated_topic}, down_arrow_is_end_marker={self.down_arrow_is_end_marker}, "
            f"treat_down_arrow_as_left={self.treat_down_arrow_as_left}, "
            f"arrow_turn_angle_rad={arrow_turn_angle:.2f}, "
            f"arrow_pass_seconds={arrow_pass_seconds:.2f}, "
            f"mission_idle_timeout_s={idle_timeout:.1f}, "
            f"complete_on_idle_timeout={complete_on_idle_timeout}"
        )

    def current_time_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 1 and msg.k[0] > 0.0:
            self.camera_fx = float(msg.k[0])

    def odom_callback(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.current_speed_mps = abs(float(msg.twist.twist.linear.x))

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # pragma: no cover - transport/runtime dependent
            self.get_logger().error(f"Failed to convert camera frame: {exc}")
            return

        arrow_candidates = self.arrow_detector.detect_candidates(frame)
        arrow_detection = self.select_arrow_candidate(arrow_candidates)
        if arrow_detection is not None:
            self.last_arrow_bbox = arrow_detection.bbox
            self.last_arrow_seen_s = self.current_time_s()
            self.reacquire_after_logo = False
            self.idle_recovery_logged = False

        logo_detection = None if self.logo_completed else self.logo_detector.detect(frame)
        if logo_detection is not None:
            method = (
                logo_detection.metadata.get("method", "unknown")
                if logo_detection.metadata
                else "unknown"
            )
            self.get_logger().info(
                "DEBUG: logo detected "
                f"bbox={logo_detection.bbox} "
                f"score={logo_detection.score:.2f} "
                f"method={method}"
            )
        moving_object_detection = self.dynamic_detector.detect(
            frame,
            focal_length_px=self.camera_fx,
            robot_speed_mps=self.current_speed_mps,
        )
        horizon_line = self.horizon_detector.detect(
            frame,
            arrow_detection.bbox if arrow_detection is not None else self.last_arrow_bbox,
        )

        self.snapshot = PerceptionSnapshot(
            arrow=arrow_detection,
            logo=logo_detection,
            moving_object=moving_object_detection,
            horizon_line=horizon_line,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            timestamp_s=self.current_time_s(),
        )

        annotated = frame.copy()
        self.draw_overlays(annotated, self.snapshot)

        try:
            overlay_message = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            overlay_message.header = msg.header
            self.annotated_image_publisher.publish(overlay_message)
        except Exception as exc:  # pragma: no cover - transport/runtime dependent
            self.get_logger().error(f"Failed to publish annotated frame: {exc}")

        if self.show_debug_window:
            cv2.imshow("ENPM673 Perception Mission", annotated)
            cv2.waitKey(1)

    def control_loop(self) -> None:
        now_s = self.current_time_s()
        if self.snapshot.frame_width == 0 or self.snapshot.frame_height == 0:
            self.publish_velocity(0.0, 0.0)
            return

        if self.mission_completed:
            self.publish_velocity(0.0, 0.0)
            return

        if not self.logo_completed and self.snapshot.logo is not None:
            self.logo_completed = True
            self.logo_pause_until_s = now_s + float(self.get_parameter("logo_pause_seconds").value)
            self.search_turn_sign = 0.0
            self.awaiting_arrow_clear = False
            self.turn_target_yaw = None
            self.turn_end_until_s = 0.0
            self.turn_command_rad_s = 0.0
            self.pending_turn_delta_rad = 0.0
            self.forward_drive_until_s = 0.0
            self.last_committed_arrow_bbox = None
            self.reacquire_after_logo = True
            self.passed_sign_recovery_until_s = 0.0
            self.publish_velocity(0.0, 0.0)
            self.get_logger().info("Task 2: UMD logo detected. Holding position for 3 seconds.")
            return

        if now_s < self.logo_pause_until_s:
            self.publish_velocity(0.0, 0.0)
            return

        if self.snapshot.moving_object is not None and self.snapshot.moving_object.metadata.get("danger", False):
            if now_s >= self.ball_stop_until_s:
                self.ball_stop_until_s = now_s + 2.0
            self.publish_velocity(0.0, 0.0)
            return

        if now_s < self.ball_stop_until_s:
            self.publish_velocity(0.0, 0.0)
            return

        if self.turn_target_yaw is not None and self.current_yaw is not None:
            yaw_error = normalize_angle(self.turn_target_yaw - self.current_yaw)
            if abs(yaw_error) > 0.10:
                turn_speed = clamp(
                    1.5 * yaw_error,
                    -float(self.get_parameter("turn_speed_rad_s").value),
                    float(self.get_parameter("turn_speed_rad_s").value),
                )
                self.publish_velocity(0.0, turn_speed)
                return
            self.turn_target_yaw = None
            self.forward_drive_until_s = now_s + float(
                self.get_parameter("post_turn_forward_seconds").value
            )
            self.get_logger().info(
                "DEBUG: turn completed. Reacquiring next arrow."
            )

        if now_s < self.turn_end_until_s:
            self.publish_velocity(0.0, self.turn_command_rad_s)
            return

        if now_s < self.forward_drive_until_s:
            self.publish_velocity(float(self.get_parameter("cruise_speed_m_s").value), 0.0)
            return

        if self.pending_turn_delta_rad != 0.0:
            if self.current_yaw is not None:
                self.turn_target_yaw = normalize_angle(
                    self.current_yaw + self.pending_turn_delta_rad
                )
                self.get_logger().info(
                    f"DEBUG: starting yaw turn. delta={self.pending_turn_delta_rad:.2f} rad"
                )
            else:
                turn_speed = float(self.get_parameter("turn_speed_rad_s").value)
                duration = abs(self.pending_turn_delta_rad) / max(0.01, turn_speed)
                self.turn_command_rad_s = turn_speed if self.pending_turn_delta_rad > 0.0 else -turn_speed
                self.turn_end_until_s = now_s + duration
                self.get_logger().info(
                    "DEBUG: starting timed turn. "
                    f"duration={duration:.2f}s angular={self.turn_command_rad_s:.2f}"
                )
            self.pending_turn_delta_rad = 0.0
            return

        if self.awaiting_arrow_clear:
            clear_hold_seconds = float(self.get_parameter("arrow_clear_hold_seconds").value)
            arrow = self.snapshot.arrow
            if arrow is not None and self.is_same_as_last_committed_arrow(arrow):
                self.coast_past_committed_arrow(arrow)
                return
            if arrow is not None and not self.is_same_as_last_committed_arrow(arrow):
                self.awaiting_arrow_clear = False
                self.last_committed_arrow_bbox = None
            else:
                if (now_s - self.last_arrow_seen_s) > clear_hold_seconds:
                    self.awaiting_arrow_clear = False
                    self.last_committed_arrow_bbox = None
                else:
                    self.coast_past_committed_arrow()
                    return

        arrow = self.snapshot.arrow
        if arrow is not None:
            self.drive_toward_arrow(arrow)
            if now_s >= self.ignore_arrow_until_s and self.should_commit_arrow(arrow):
                self.commit_arrow_action(arrow)
            return

        idle_timeout = float(self.get_parameter("mission_idle_timeout_s").value)
        if self.executed_arrow_count > 0 and (now_s - self.last_arrow_seen_s) > idle_timeout:
            if bool(self.get_parameter("complete_on_idle_timeout").value):
                self.mission_completed = True
                self.publish_velocity(0.0, 0.0)
                self.get_logger().info("Mission complete: no new arrow detected after the final maneuver.")
                return

            if not self.idle_recovery_logged:
                self.get_logger().info(
                    "Task 1: no new arrow detected yet. Resetting search state and continuing."
                )
                self.idle_recovery_logged = True

            self.awaiting_arrow_clear = False
            self.last_committed_arrow_bbox = None
            self.ignore_arrow_until_s = 0.0
            self.forward_drive_until_s = 0.0
            self.turn_target_yaw = None
            self.turn_end_until_s = 0.0
            self.turn_command_rad_s = 0.0
            self.pending_turn_delta_rad = 0.0
            self.search_turn_sign = 0.0
            self.reacquire_after_logo = False
            self.search_for_next_arrow(now_s)
            return

        self.search_for_next_arrow(now_s)

    def drive_toward_arrow(self, arrow: DetectionResult) -> None:
        width = float(self.snapshot.frame_width)
        center_x = float(arrow.bbox.center[0])
        normalized_error = (center_x - width / 2.0) / max(1.0, width / 2.0)
        gain = float(self.get_parameter("centering_gain").value)
        angular = clamp(-gain * normalized_error, -1.0, 1.0)
        linear = float(self.get_parameter("cruise_speed_m_s").value)
        linear *= max(0.30, 1.0 - abs(normalized_error))
        self.publish_velocity(linear, angular)

    def coast_past_committed_arrow(self, arrow: DetectionResult | None = None) -> None:
        linear = 0.75 * float(self.get_parameter("cruise_speed_m_s").value)
        angular = 0.0
        if arrow is not None:
            width = float(self.snapshot.frame_width)
            center_x = float(arrow.bbox.center[0])
            normalized_error = (center_x - width / 2.0) / max(1.0, width / 2.0)
            # Keep only a very gentle heading correction while clearing a sign
            # the controller has already committed to, otherwise the robot can
            # curve back toward a stale floor arrow instead of moving on.
            angular = clamp(-0.22 * normalized_error, -0.12, 0.12)
        self.publish_velocity(linear, angular)

    def should_ignore_passed_down_arrow(self, arrow: DetectionResult) -> bool:
        if arrow.direction != "down" or self.down_arrow_is_end_marker:
            return False
        frame_height = float(self.snapshot.frame_height)
        frame_width = float(self.snapshot.frame_width)
        area_ratio = arrow.bbox.area / max(1.0, frame_width * frame_height)
        low_in_frame = arrow.bbox.center[1] > 0.68 * frame_height
        close_enough = arrow.bbox.bottom > 0.84 * frame_height or area_ratio > 0.09
        return self.is_same_as_last_committed_arrow(arrow) or (low_in_frame and close_enough)

    def should_commit_arrow(self, arrow: DetectionResult) -> bool:
        frame_width = float(self.snapshot.frame_width)
        frame_height = float(self.snapshot.frame_height)
        area_ratio = arrow.bbox.area / max(1.0, frame_width * frame_height)
        center_ratio = float(self.get_parameter("arrow_commit_center_ratio").value)
        bottom_ratio = float(self.get_parameter("arrow_commit_bottom_ratio").value)
        area_threshold = float(self.get_parameter("arrow_commit_area_ratio").value)
        centered = abs(arrow.bbox.center[0] - frame_width / 2.0) < center_ratio * frame_width
        close_enough = arrow.bbox.bottom > bottom_ratio * frame_height or area_ratio > area_threshold
        return centered and close_enough

    def commit_arrow_action(self, arrow: DetectionResult) -> None:
        now_s = self.current_time_s()
        direction = arrow.direction or "up"
        self.get_logger().info(
            "DEBUG: committing arrow "
            f"direction={direction} center={arrow.bbox.center} "
            f"bottom={arrow.bbox.bottom} area={arrow.bbox.area}"
        )

        if direction == "down" and self.down_arrow_is_end_marker:
            self.mission_completed = True
            self.publish_velocity(0.0, 0.0)
            self.get_logger().info("Task 1: end marker reached. Stopping mission.")
            return

        self.executed_arrow_count += 1
        self.ignore_arrow_until_s = now_s + float(self.get_parameter("arrow_ignore_seconds").value)
        self.awaiting_arrow_clear = True
        self.forward_drive_until_s = now_s + float(self.get_parameter("arrow_pass_seconds").value)
        self.turn_target_yaw = None
        self.turn_end_until_s = 0.0
        self.turn_command_rad_s = 0.0
        turn_angle = float(self.get_parameter("arrow_turn_angle_rad").value)
        if direction == "left":
            self.pending_turn_delta_rad = turn_angle
        elif direction == "right":
            self.pending_turn_delta_rad = -turn_angle
        elif direction == "down":
            if self.treat_down_arrow_as_left:
                # Temporary sim fix: detector often sees the floor arrow as down.
                # Treat it as a left turn so the robot continues around the path.
                self.pending_turn_delta_rad = turn_angle
            else:
                self.pending_turn_delta_rad = 0.0
        else:
            # up / straight arrow
            self.pending_turn_delta_rad = 0.0
        self.last_committed_arrow_bbox = arrow.bbox
        self.passed_sign_recovery_until_s = 0.0
        # In the floor-arrow simulation, the path is already encoded by the
        # visible arrow placements. Keeping a persistent left/right bias after a
        # sign is passed tends to pull the robot off the arc, so reacquire the
        # next sign neutrally.
        self.search_turn_sign = 0.0

        if direction == "down":
            self.get_logger().info("Task 1: down arrow accepted.")
        elif direction == "up":
            self.get_logger().info("Task 1: straight arrow accepted.")
        else:
            self.get_logger().info(f"Task 1: {direction} arrow accepted.")

    def select_arrow_candidate(
        self, candidates: list[DetectionResult]
    ) -> DetectionResult | None:
        if not candidates:
            return None

        if not self.awaiting_arrow_clear or self.last_committed_arrow_bbox is None:
            return candidates[0]

        for candidate in candidates:
            if not self.is_same_as_last_committed_arrow(candidate):
                return candidate

        return candidates[0]

    def is_same_as_last_committed_arrow(self, arrow: DetectionResult) -> bool:
        if self.last_committed_arrow_bbox is None:
            return False
        previous = self.last_committed_arrow_bbox
        overlap = self.bbox_iou(previous, arrow.bbox)
        dx = float(previous.center[0] - arrow.bbox.center[0])
        dy = float(previous.center[1] - arrow.bbox.center[1])
        center_distance = float(np.hypot(dx, dy))
        scale = max(
            1.0,
            float(previous.width + previous.height + arrow.bbox.width + arrow.bbox.height) / 4.0,
        )
        return overlap > 0.12 or center_distance < 0.95 * scale

    def bbox_iou(self, first: BoundingBox, second: BoundingBox) -> float:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        inter_width = max(0, right - left)
        inter_height = max(0, bottom - top)
        intersection = float(inter_width * inter_height)
        union = float(first.area + second.area) - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def search_for_next_arrow(self, now_s: float) -> None:
        search_speed = float(self.get_parameter("search_speed_m_s").value)
        search_turn_speed = float(self.get_parameter("search_turn_speed_rad_s").value)

        if self.executed_arrow_count == 0 and self.last_arrow_seen_s == 0.0:
            angular = search_turn_speed if int(now_s) % 6 < 3 else -search_turn_speed
            self.publish_velocity(0.0, angular)
            return

        if self.reacquire_after_logo:
            angular = search_turn_speed if int(now_s) % 6 < 3 else -search_turn_speed
            self.publish_velocity(0.0, angular)
            return

        if self.search_turn_sign != 0.0:
            self.publish_velocity(
                0.0,
                self.search_turn_sign * 0.55 * search_turn_speed,
            )
            return

        angular = 0.28 * search_turn_speed if int(now_s) % 4 < 2 else -0.28 * search_turn_speed
        if self.executed_arrow_count > 0:
            self.publish_velocity(0.0, angular)
            return

        self.publish_velocity(0.65 * search_speed, angular)

    def publish_velocity(self, linear_x: float, angular_z: float) -> None:
        if self.cmd_vel_stamped:
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = self.cmd_vel_frame_id
            twist.twist.linear.x = float(linear_x)
            twist.twist.angular.z = float(angular_z)
            self.cmd_publisher.publish(twist)
            return

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_publisher.publish(twist)

    def draw_box(
        self,
        frame,
        detection: DetectionResult,
        color: tuple[int, int, int],
        title: str,
        extra_text: str | None = None,
    ) -> None:
        x, y, width, height = (
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.width,
            detection.bbox.height,
        )
        if detection.quad:
            quad = np.array(detection.quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [quad], isClosed=True, color=color, thickness=2)
        else:
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        label = title if extra_text is None else f"{title} | {extra_text}"
        cv2.putText(
            frame,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_overlays(self, frame, snapshot: PerceptionSnapshot) -> None:
        if snapshot.horizon_line is not None:
            x1, y1, x2, y2 = snapshot.horizon_line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(
                frame,
                "HORIZON",
                (10, max(25, min(y1, y2) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        if snapshot.arrow is not None:
            direction_text = {
                "up": "STRAIGHT",
                "left": "LEFT",
                "right": "RIGHT",
                "down": "END" if self.down_arrow_is_end_marker else "DOWN",
            }.get(snapshot.arrow.direction or "", "UNKNOWN")
            self.draw_box(
                frame,
                snapshot.arrow,
                (0, 255, 0),
                f"ARROW: {direction_text}",
            )

        if snapshot.logo is not None:
            self.draw_box(frame, snapshot.logo, (0, 0, 255), "UMD LOGO")

        if snapshot.moving_object is not None:
            x, y, width, height = (
                snapshot.moving_object.bbox.x,
                snapshot.moving_object.bbox.y,
                snapshot.moving_object.bbox.width,
                snapshot.moving_object.bbox.height,
            )
            label_parts = ["MOVING"]
            if snapshot.moving_object.ttc_seconds is not None:
                label_parts.append(f"TTC {snapshot.moving_object.ttc_seconds:.2f}s")
            elif snapshot.moving_object.distance_m is not None:
                label_parts.append(f"{snapshot.moving_object.distance_m:.2f}m")
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 2)
            cv2.putText(
                frame,
                " | ".join(label_parts),
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        state = "RUNNING"
        now_s = self.current_time_s()
        if self.mission_completed:
            state = "COMPLETE"
        elif now_s < self.logo_pause_until_s:
            state = "LOGO STOP"
        elif now_s < self.ball_stop_until_s:
            state = "BALL STOP"
        elif self.turn_target_yaw is not None or now_s < self.turn_end_until_s:
            state = "TURNING"
        elif now_s < self.forward_drive_until_s:
            state = "ADVANCING"
        elif snapshot.arrow is None:
            state = "SEARCHING"

        cv2.putText(
            frame,
            f"STATE: {state}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_velocity(0.0, 0.0)
        except Exception:
            pass

        try:
            if node.show_debug_window:
                cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
