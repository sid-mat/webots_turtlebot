import os
import launch
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from webots_ros2_driver.urdf_spawner import URDFSpawner, get_webots_driver_node
from webots_ros2_driver.webots_launcher import WebotsLauncher


def get_ros2_nodes(*args):
    package_dir = get_package_share_directory('tb4_sim')
    tb4_xacro_path = os.path.join(package_dir, 'resource', 'tb4_webots.xacro')
    rviz_config_path = os.path.join(package_dir, 'rviz', 'perception_demo.rviz')
    tb4_description = xacro.process_file(
        tb4_xacro_path,
        mappings={'name': 'turtlebot4'}
    ).toxml()
    ros2_control_params = os.path.join(package_dir, 'resource', 'tb4_control.yaml')

    spawn_URDF_tb4 = URDFSpawner(
        name='turtlebot4',
        robot_description=tb4_description,
        relative_path_prefix=os.path.join(package_dir, 'resource'),
        translation='6.66 0.327 -0.00564',
        # Face the first arrow sign at startup instead of spawning sideways.
        rotation='0 0 1 -1.57',
    )

    mappings = [('/diffdrive_controller/cmd_vel_unstamped', '/cmd_vel')]
    if os.environ.get('ROS_DISTRO') in ['humble', 'rolling']:
        mappings.append(('/diffdrive_controller/odom', '/odom'))

    tb4_driver = Node(
        package='webots_ros2_driver',
        executable='driver',
        output='screen',
        additional_env={'WEBOTS_CONTROLLER_URL': 'turtlebot4'},
        parameters=[
            {
                'robot_description': tb4_description,
                'use_sim_time': True,
                'set_robot_state_publisher': True,
            },
            ros2_control_params,
        ],
        remappings=mappings,
    )

    mission_controller = Node(
        package='tb4_sim',
        executable='mission_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'show_debug_window': True,
            # Use explicit simulation tuning so launch behavior is predictable.
            'turn_speed_rad_s': 0.85,
            'cruise_speed_m_s': 0.14,
            'centering_gain': 1.55,
            'arrow_turn_angle_rad': 1.15,
            'arrow_commit_center_ratio': 0.30,
            'arrow_commit_bottom_ratio': 0.70,
            'arrow_commit_area_ratio': 0.04,
            'arrow_pass_seconds': 0.15,
            'post_turn_forward_seconds': 0.20,
            'mission_idle_timeout_s': 9999.0,
            'complete_on_idle_timeout': False,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    # Ball robot extern controller
    ball_robot_driver = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(
                get_package_share_directory('tb4_sim'),
                'controllers', 'ball_robot', 'ball_robot.py'
            )
        ],
        additional_env={
            'WEBOTS_CONTROLLER_URL': 'ipc://1234/ball_robot',
            'WEBOTS_HOME': '/usr/local/webots',
            'LD_LIBRARY_PATH': '/usr/local/webots/lib/controller',
        },
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }],
    )

    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
    )

    load_jsb = TimerAction(period=10.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             '/controller_manager/load_controller',
             'controller_manager_msgs/srv/LoadController',
             "{name: 'joint_state_broadcaster'}"],
        output='screen',
    )])

    configure_jsb = TimerAction(period=12.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             '/controller_manager/configure_controller',
             'controller_manager_msgs/srv/ConfigureController',
             "{name: 'joint_state_broadcaster'}"],
        output='screen',
    )])

    load_diff = TimerAction(period=14.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             '/controller_manager/load_controller',
             'controller_manager_msgs/srv/LoadController',
             "{name: 'diffdrive_controller'}"],
        output='screen',
    )])

    configure_diff = TimerAction(period=16.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             '/controller_manager/configure_controller',
             'controller_manager_msgs/srv/ConfigureController',
             "{name: 'diffdrive_controller'}"],
        output='screen',
    )])

    activate_both = TimerAction(period=18.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call',
             '/controller_manager/switch_controller',
             'controller_manager_msgs/srv/SwitchController',
             "{activate_controllers: ['joint_state_broadcaster', 'diffdrive_controller'], "
             "deactivate_controllers: [], strictness: 1}"],
        output='screen',
    )])

    return [
        spawn_URDF_tb4,
        robot_state_publisher,
        footprint_publisher,
        ball_robot_driver,
        load_jsb,
        configure_jsb,
        load_diff,
        configure_diff,
        activate_both,
        mission_controller,
        rviz,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessIO(
                target_action=spawn_URDF_tb4,
                on_stdout=lambda event: get_webots_driver_node(event, [tb4_driver]),
            )
        ),
    ]


def launch_webots(context, *args, **kwargs):
    package_share = get_package_share_directory('tb4_sim')
    world_file = LaunchConfiguration('world').perform(context)
    world_path = os.path.join(package_share, 'worlds', world_file)

    webots = WebotsLauncher(
        world=world_path,
        ros2_supervisor=True,
    )

    return [
        webots,
        webots._supervisor,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ]



def generate_launch_description():
    # Ensure controller plugins are findable
    os.environ['LD_LIBRARY_PATH'] = (
        '/opt/ros/humble/lib:'
        + os.environ.get('LD_LIBRARY_PATH', '')
    )
    os.environ['AMENT_PREFIX_PATH'] = (
        '/opt/ros/humble:'
        + os.environ.get('AMENT_PREFIX_PATH', '')
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='house.wbt',
            description='Choose one of the world files from the tb4_sim share directory'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz with the annotated perception feed'
        ),
        OpaqueFunction(function=launch_webots),
    ] + get_ros2_nodes())
