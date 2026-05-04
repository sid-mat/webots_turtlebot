from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='enpm673_final',
            executable='final_node',
            name='final_project_node',
            output='screen'
        )
    ])