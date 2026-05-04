#!/usr/bin/env python3
"""
Oscillating ball controller - runs as standalone Webots extern controller.
Connect via WEBOTS_CONTROLLER_URL=ball_robot
"""
import sys
import math
import os

# Add Webots Python library to path
webots_home = os.environ.get('WEBOTS_HOME', '/usr/local/webots')
sys.path.insert(0, os.path.join(webots_home, 'lib', 'controller', 'python'))

from controller import Robot

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    motor = robot.getDevice('ball_motor')
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

    t = 0.0
    AMPLITUDE = 0.2
    FREQUENCY = 0.5

    while robot.step(timestep) != -1:
        t += timestep / 1000.0
        vel = AMPLITUDE * FREQUENCY * 2 * math.pi * math.cos(
            2 * math.pi * FREQUENCY * t
        )
        motor.setVelocity(vel)

if __name__ == '__main__':
    main()