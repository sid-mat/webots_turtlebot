#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = """
/cmd_vel teleop
----------------------
i : forward
, : backward
j : turn left
l : turn right
u : forward + left
o : forward + right
m : backward + right
. : backward + left
k : stop
q : quit
"""


class SimpleTeleop(Node):
    def __init__(self):
        super().__init__('simple_cmd_vel_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear_speed = 1.0
        self.angular_speed = 0.3

    def send(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub.publish(msg)

    def stop(self) -> None:
        self.send(0.0, 0.0)


def get_key(timeout: float = 0.1) -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if readable:
            return sys.stdin.read(1)
        return ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rclpy.init()
    node = SimpleTeleop()

    keymap = {
        'i': ( node.linear_speed,  0.0),
        ',': (-node.linear_speed,  0.0),
        'j': ( 0.0,  node.angular_speed),
        'l': ( 0.0, -node.angular_speed),
        'u': ( node.linear_speed,  node.angular_speed),
        'o': ( node.linear_speed, -node.angular_speed),
        'm': (-node.linear_speed, -node.angular_speed),
        '.': (-node.linear_speed,  node.angular_speed),
        'k': ( 0.0,  0.0),
    }

    print(HELP)

    try:
        while rclpy.ok():
            key = get_key()

            if key == 'q':
                break

            if key in keymap:
                linear_x, angular_z = keymap[key]
                node.send(linear_x, angular_z)
                print(f"sent: linear.x={linear_x:.2f}, angular.z={angular_z:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()