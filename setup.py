from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lidar_object_filter'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Simon',
    maintainer_email='simon.teveshumme@gmail.com',
    description='Removes dynamic actors from LiDAR point clouds using ZED 3D bounding boxes',
    license='MIT',
    entry_points={
        'console_scripts': [
            'filter_node = lidar_object_filter.filter_node:main',
        ],
    },
)
