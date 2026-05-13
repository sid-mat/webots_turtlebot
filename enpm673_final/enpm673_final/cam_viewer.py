
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
class CompressedCameraViewer(Node):
    def __init__(self):
        super().__init__('tb4_compressed_camera_viewer')
        self.image_topic = '/tb4_5/oakd/rgb/preview/image_raw/compressed'
        self.subscription = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f'Subscribed to {self.image_topic}')

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode compressed image')
            return
        
        cv2.imshow('TB4 OAK-D RGB Compressed Camera', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.get_logger().info('Closing camera viewer')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CompressedCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
