import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DiagNode(Node):
    def __init__(self):
        super().__init__('diag_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/camera/image_raw/image_color', self.cb, 10)

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]

        # Crop bottom 40%
        roi = frame[int(h*0.6):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Show multiple threshold levels side by side
        _, t120 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        _, t150 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        _, t180 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        _, t200 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Stack them for comparison
        row1 = np.hstack([
            cv2.cvtColor(t120, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(t150, cv2.COLOR_GRAY2BGR)
        ])
        row2 = np.hstack([
            cv2.cvtColor(t180, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(t200, cv2.COLOR_GRAY2BGR)
        ])
        combined = np.vstack([row1, row2])

        cv2.putText(combined, "t120", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(combined, "t150", (w//2+10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(combined, "t180", (10, row1.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(combined, "t200", (w//2+10, row1.shape[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("raw", roi)
        cv2.imshow("thresholds", combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DiagNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()