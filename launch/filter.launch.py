import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(
        get_package_share_directory('lidar_object_filter'), 'config')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz_original',
            arguments=['-d', os.path.join(cfg, 'lidar_filter_original_only.rviz')],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz_filtered',
            arguments=['-d', os.path.join(cfg, 'lidar_filter_filtered_only.rviz')],
            output='screen',
        ),
        Node(
            package='lidar_object_filter',
            executable='filter_node',
            name='dynamic_object_filter',
            output='screen',
            parameters=[{
                'lidar_topic':    '/microvision/pointcloud',
                'objects_topic':  '/zed/zed_node/obj_det/objects',
                'output_topic':   '/microvision/pointcloud/filtered',
                'padding':          0.2,   # extra margin around each box [m]
                'max_object_age':   1.0,   # ignore objects older than this [s]
                'min_confidence':   0.4,   # ignore detections below this confidence
                'objects_tf_frame': '',
                # Fixed camera→lidar offset from URDF (bypasses TF).
                # LiDAR: xyz=(0.15, 0, 0.61)  Camera: xyz=(0.20, 0, 0.68) pitch=0.15 rad
                # P_lidar = R_pitch(0.15) @ P_camera + (0.05, 0, 0.07)
                'fixed_transform_x':     0.05,
                'fixed_transform_y':     0.0,
                'fixed_transform_z':     0.07,
                'fixed_transform_roll':  0.0,
                'fixed_transform_pitch': 0.15,
                'fixed_transform_yaw':   0.0,
            }],
        ),
    ])
