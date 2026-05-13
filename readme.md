# ENPM673 Final Project — TurtleBot4 Perception & Navigation

ROS 2 package for a TurtleBot4 perception pipeline running in Webots simulation or on real hardware. Completes four vision-based navigation tasks using a single camera feed.

## Tasks

| # | Task | Method | Points |
|---|------|--------|--------|
| 1 | Arrow paper following | YOLO (`best.pt`) + proportional angular control | 25 |
| 2 | UMD logo detection & 3s stop | ORB feature matching + homography | 25 |
| 3 | Moving ball detection + TTC | MOG2 background subtraction + circularity filter | 25 |
| 4 | Horizon detection | RANSAC line fitting | 25 |

## Package Structure

```
webots_turtlebot-master/
└── enpm673_final/
    ├── enpm673_final/
    │   ├── perception_node.py   # main ROS 2 node + all detectors
    │   ├── camera.py
    │   └── cam_viewer.py
    ├── launch/
    │   └── launch.py
    ├── package.xml
    └── setup.py
```

## Prerequisites

- ROS 2 (tested on Humble)
- Python packages: `opencv-python`, `numpy`, `ultralytics`
- ROS packages: `rclpy`, `sensor_msgs`, `geometry_msgs`, `cv_bridge`

## Required Assets

Place these in `enpm673_final/enpm673_final/` before building:

- `best.pt` — YOLO model for arrow detection
- `assets/umd_logo.png` — UMD logo template for Task 2

## Build & Run

```bash
# Build
cd <your_ws>
colcon build --packages-select enpm673_final
source install/setup.bash

# Launch (real robot, tb4_5 namespace)
ros2 launch enpm673_final launch.py

# Flags
ros2 run enpm673_final final_node --sim      # use sim camera topic
ros2 run enpm673_final final_node --real     # use OAK-D topic (default)
ros2 run enpm673_final final_node --no-nav   # perception only, no cmd_vel
```

## Standalone / Debug (no ROS 2 required)

```bash
cd enpm673_final/enpm673_final

# Run all modules on webcam (id 0) or a video file
python3 perception_node.py --test 0
python3 perception_node.py --test video.mp4

# Stress test a single module
python3 perception_node.py --stress arrow   0
python3 perception_node.py --stress umd     0
python3 perception_node.py --stress ball    0
python3 perception_node.py --stress horizon 0
```

## Topics

| Topic | Type | Direction |
|-------|------|-----------|
| `/{TB}/oakd/rgb/preview/image_raw/compressed` | `sensor_msgs/CompressedImage` | sub |
| `/{TB}/cmd_vel` | `geometry_msgs/Twist` | pub |
| `/{TB}/odom` | `nav_msgs/Odometry` | sub |
| `/{TB}/enpm673/perception_viz` | `sensor_msgs/Image` | pub |

Default namespace: `tb4_5`. Change `TB_NUMBER` at the top of `perception_node.py`.

## Key Config (perception_node.py → `Config` class)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FORWARD_SPEED` | 0.10 m/s | Base forward speed |
| `CENTERING_KP` | 0.45 | P-gain for arrow centering |
| `ARROW_CONF` | 0.20 | YOLO confidence threshold |
| `UMD_STOP_DURATION` | 3.0 s | Stop duration on logo detection |
| `UMD_COOLDOWN_S` | 8.0 s | Refractory period after UMD stop |
