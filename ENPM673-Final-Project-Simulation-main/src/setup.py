"""webots_ros2 package setup file."""

from glob import glob
from setuptools import find_packages, setup

package_name = 'tb4_sim'

data_files = []
data_files.append((
    'share/ament_index/resource_index/packages',
    ['resource/' + package_name]
))
data_files.append((
    'share/' + package_name,
    ['package.xml']
))
data_files.append((
    'share/' + package_name + '/launch',
    ['launch/tb4_launcher.py']
))
data_files.append((
    'share/' + package_name + '/resource',
    [
        'resource/tb4_raw.urdf',
        'resource/tb4_webots.xacro',
        'resource/tb4_control.yaml',
        'resource/helper.xacro'
    ]
))
data_files.append((
    'share/' + package_name + '/controllers/ball_robot',
    ['controllers/ball_robot/ball_robot.py']
))
data_files.append((
    'share/' + package_name + '/worlds',
    [
        'worlds/house.wbt',
        'worlds/mini_office.wbt',
        'worlds/office.wbt',
    ]
))
data_files.append((
    'share/' + package_name + '/textures',
    glob('textures/*')
))
data_files.append((
    'share/' + package_name + '/rviz',
    ['rviz/perception_demo.rviz']
))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=data_files,
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    author='CT Ingenieros',
    author_email='oscar.martinez@ctingenieros.es',
    maintainer='Óscar Martínez',
    maintainer_email='oscar.martinez@ctingenieros.es',
    keywords=['ROS', 'Webots', 'Robot', 'Simulation', 'Examples'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='TurtleBot4 robot ROS2 interface for Webots.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_controller = tb4_sim.mission_controller:main',
        ],
        'launch.frontend.launch_extension': ['launch_ros = launch_ros']
    }
)
