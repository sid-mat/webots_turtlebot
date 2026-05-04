import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time

from enpm673_final.task1_arrow import ArrowDetector
from enpm673_final.task2_logo import LogoDetector
from enpm673_final.task3_ball import BallDetector
from enpm673_final.task4_horizon import HorizonDetector

class FinalProjectNode(Node):

    STATE_SEARCHING = "SEARCHING"
    STATE_COOLDOWN  = "COOLDOWN"
    STATE_LOGO_STOP = "LOGO_STOP"
    STATE_BALL_STOP = "BALL_STOP"
    STATE_DONE      = "DONE"

    def __init__(self):
        super().__init__('final_project_node')
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, '/camera/image_raw/image_color', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.arrow   = ArrowDetector()
        self.logo    = LogoDetector()
        self.ball    = BallDetector()
        self.horizon = HorizonDetector()

        self.state       = self.STATE_SEARCHING
        self.state_start = time.time()

        self.DRIVE_SPEED   = 0.12
        self.LOGO_HOLD     = 3.0
        self.COOLDOWN_TIME = 1.5

        self.get_logger().info("=== Final Project Node Started ===")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        arrow_angle, arrow_offset, arrow_bbox, frame, path_done = self.arrow.process(frame)
        logo_detected, frame      = self.logo.process(frame)
        ball_blocking, ttc, frame = self.ball.process(frame)
        frame                     = self.horizon.process(frame, arrow_bbox)

        if path_done and self.state != self.STATE_DONE:
            self.get_logger().info("Path complete — stopping!")
            self.transition(self.STATE_DONE)

        self.update_state(arrow_angle, logo_detected, ball_blocking)

        cv2.putText(frame, f"STATE: {self.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if arrow_angle is not None:
            cv2.putText(frame, f"OFFSET: {arrow_angle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"NO ARROW: {self.arrow.no_arrow_count}/{self.arrow.NO_ARROW_LIMIT}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        frame = cv2.resize(frame, (640, 360))
        cv2.imshow("ENPM673 Final Project", frame)
        cv2.waitKey(1)

    def update_state(self, arrow_angle, logo_detected, ball_blocking):
        now     = time.time()
        elapsed = now - self.state_start

        # Priority 1: Ball
        if ball_blocking and self.state != self.STATE_DONE:
            if self.state != self.STATE_BALL_STOP:
                self.transition(self.STATE_BALL_STOP)
            self.stop()
            return

        # Priority 2: Logo
        if logo_detected and self.state not in \
                [self.STATE_LOGO_STOP, self.STATE_DONE]:
            self.transition(self.STATE_LOGO_STOP)

        if self.state == self.STATE_LOGO_STOP:
            self.stop()
            if elapsed > self.LOGO_HOLD:
                self.transition(self.STATE_SEARCHING)
            return

        if self.state == self.STATE_BALL_STOP:
            self.stop()
            self.transition(self.STATE_SEARCHING)
            return

        if self.state == self.STATE_DONE:
            self.stop()
            return

        if self.state == self.STATE_COOLDOWN:
            self.drive(linear=self.DRIVE_SPEED)
            if elapsed > self.COOLDOWN_TIME:
                self.transition(self.STATE_SEARCHING)
            return
    
        # No valid arrow visible
        if arrow_angle is None:
            self.drive(linear=self.DRIVE_SPEED)  # full speed straight, not 0.5
            return

        offset   = arrow_angle
        KP       = 0.3
        DEADZONE = 0.08

        if abs(offset) < DEADZONE:
            angular = 0.0
        else:
            angular = -offset * KP
            angular = max(-0.3, min(0.3, angular))

        self.drive(linear=self.DRIVE_SPEED, angular=angular)

    def transition(self, new_state):
        self.state       = new_state
        self.state_start = time.time()

    def drive(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x  = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.drive(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = FinalProjectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()