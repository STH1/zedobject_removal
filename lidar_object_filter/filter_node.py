#!/usr/bin/env python3
"""
Removes dynamic actors detected by the ZED camera from MicroVision LiDAR point clouds.

Subscribes to:
  /microvision/pointcloud          – sensor_msgs/PointCloud2
  /zed/zed_node/obj_det/objects    – zed_msgs/ObjectsStamped
  /tf_static                       – tf2_msgs/TFMessage  (TRANSIENT_LOCAL, cached)
  /tf                              – tf2_msgs/TFMessage  (latest per edge, cached)

Publishes:
  /microvision/pointcloud/filtered          – sensor_msgs/PointCloud2
  /dynamic_object_filter/boxes             – visualization_msgs/MarkerArray
    Bounding boxes after transformation into the LiDAR frame.
    Visualise in RViz with Fixed Frame = MicroVisionScanTF.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from zed_msgs.msg import ObjectsStamped


_QOS_SENSOR = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)
_QOS_TF_STATIC = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=100,
)
_QOS_TF = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=100,
)

# ZED BoundingBox3D corner ordering (top face 0-3, bottom face 4-7):
#   3──2   7──6
#   │  │   │  │
#   0──1   4──5
_AXIS_PAIRS = ((0, 1), (0, 3), (0, 4))

# 12 edges of the box as corner-index pairs
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # top face
    (4, 5), (5, 6), (6, 7), (7, 4),   # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]

# Colour palette per label (RGB 0-1)
_LABEL_COLORS: dict[str, tuple] = {
    'Person':     (1.0, 0.2, 0.2),
    'Vehicle':    (0.2, 0.4, 1.0),
    'Animal':     (1.0, 0.7, 0.0),
    'Bag':        (0.6, 0.0, 0.8),
    'Electronics':(0.0, 0.8, 0.8),
    'default':    (0.0, 1.0, 0.4),
}


# ── Math helpers ──────────────────────────────────────────────────────────────

def _quat_to_rot(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _rpy_to_rot(roll, pitch, yaw) -> np.ndarray:
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1,  0,   0 ], [0, cr, -sr], [0,  sr, cr]])
    Ry = np.array([[cp, 0,   sp], [0,  1,   0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0 ], [sy, cy,  0], [0,   0,  1]])
    return Rz @ Ry @ Rx


def _tf_to_Rt(transform):
    q = transform.rotation
    t = transform.translation
    return _quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


def _extract_xyz(cloud: PointCloud2) -> np.ndarray:
    n = cloud.width * cloud.height
    step = cloud.point_step
    field_map = {f.name: f.offset for f in cloud.fields}
    raw = np.frombuffer(cloud.data, dtype=np.uint8).reshape(n, step)
    xyz = np.empty((n, 3), dtype=np.float32)
    for col, name in enumerate(('x', 'y', 'z')):
        off = field_map[name]
        xyz[:, col] = raw[:, off:off + 4].copy().view(np.float32).reshape(-1)
    return xyz


def _build_obb(corners: np.ndarray, padding: float):
    center = corners.mean(axis=0)
    axes, half_extents = [], []
    for i, j in _AXIS_PAIRS:
        edge = corners[j] - corners[i]
        length = np.linalg.norm(edge)
        if length < 1e-4:
            return None
        axes.append(edge / length)
        half_extents.append(length / 2.0 + padding)
    return center, np.array(axes), np.array(half_extents)


def _points_in_obb(xyz: np.ndarray, center, axes, half_extents) -> np.ndarray:
    diffs = xyz.astype(np.float64) - center
    inside = np.ones(len(xyz), dtype=bool)
    for axis, he in zip(axes, half_extents):
        inside &= np.abs(diffs @ axis) <= he
    return inside


# ── TF cache with ancestor-chain lookup ──────────────────────────────────────

class TFCache:
    """
    Flat edge store: child_frame -> (parent_frame, R_parent_child, t_parent_child)
    where  P_parent = R @ P_child + t.
    """

    def __init__(self):
        self._edges: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

    def update(self, msg: TFMessage) -> None:
        for tf in msg.transforms:
            R, t = _tf_to_Rt(tf.transform)
            self._edges[tf.child_frame_id] = (tf.header.frame_id, R, t)

    def lookup(self, target: str, source: str):
        """Return (R, t) s.t. P_target = R @ P_source + t. Raises LookupError."""
        src_chain = self._chain_to_root(source)
        tgt_chain = self._chain_to_root(target)
        for ancestor, (R_anc_tgt, t_anc_tgt) in tgt_chain.items():
            if ancestor in src_chain:
                R_anc_src, t_anc_src = src_chain[ancestor]
                R = R_anc_tgt.T @ R_anc_src
                t = R_anc_tgt.T @ (t_anc_src - t_anc_tgt)
                return R, t
        known = sorted(set(src_chain) | set(tgt_chain))
        raise LookupError(f'No path "{source}" → "{target}". Known: {known}')

    def _chain_to_root(self, frame: str) -> dict:
        result = {frame: (np.eye(3), np.zeros(3))}
        R_acc, t_acc = np.eye(3), np.zeros(3)
        current = frame
        for _ in range(32):
            if current not in self._edges:
                break
            parent, R_par_cur, t_par_cur = self._edges[current]
            t_acc = R_par_cur @ t_acc + t_par_cur
            R_acc = R_par_cur @ R_acc
            if parent in result:
                break
            result[parent] = (R_acc.copy(), t_acc.copy())
            current = parent
        return result

    @property
    def known_frames(self):
        frames = set(self._edges.keys())
        for parent, _, _ in self._edges.values():
            frames.add(parent)
        return sorted(frames)


# ── Marker helpers ────────────────────────────────────────────────────────────

def _box_marker(marker_id: int, corners_lidar: np.ndarray, label: str,
                frame_id: str, stamp) -> Marker:
    """LINE_LIST marker for one 3D bounding box (12 edges)."""
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp    = stamp
    m.ns              = 'boxes'
    m.id              = marker_id
    m.type            = Marker.LINE_LIST
    m.action          = Marker.ADD
    m.scale.x         = 0.05          # line width [m]
    m.lifetime.sec    = 1             # auto-delete after 1 s

    r, g, b = _LABEL_COLORS.get(label, _LABEL_COLORS['default'])
    m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)

    for i, j in _BOX_EDGES:
        for idx in (i, j):
            p = Point()
            p.x, p.y, p.z = float(corners_lidar[idx, 0]), \
                             float(corners_lidar[idx, 1]), \
                             float(corners_lidar[idx, 2])
            m.points.append(p)
    return m


def _text_marker(marker_id: int, center: np.ndarray, label: str,
                 confidence: float, frame_id: str, stamp) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp    = stamp
    m.ns              = 'labels'
    m.id              = marker_id
    m.type            = Marker.TEXT_VIEW_FACING
    m.action          = Marker.ADD
    m.pose.position.x = float(center[0])
    m.pose.position.y = float(center[1])
    m.pose.position.z = float(center[2]) + 0.3
    m.pose.orientation.w = 1.0
    m.scale.z         = 0.3
    m.color           = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
    m.text            = f'{label} {confidence:.0%}'
    m.lifetime.sec    = 1
    return m


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class DynamicObjectFilter(Node):

    def __init__(self):
        super().__init__('dynamic_object_filter')

        self.declare_parameter('lidar_topic',     '/microvision/pointcloud')
        self.declare_parameter('objects_topic',   '/zed/zed_node/obj_det/objects')
        self.declare_parameter('output_topic',    '/microvision/pointcloud/filtered')
        self.declare_parameter('marker_topic',    '/dynamic_object_filter/boxes')
        self.declare_parameter('padding',          0.2)
        self.declare_parameter('max_object_age',   1.0)
        self.declare_parameter('min_confidence',   0.4)
        # Leave empty to use the frame_id from the objects message header.
        self.declare_parameter('objects_tf_frame', '')
        # Fixed camera→lidar transform (bypasses TF entirely).
        # Set all six to use a hardcoded offset instead of TF lookup.
        # P_lidar = R(roll,pitch,yaw) @ P_camera + (x, y, z)
        self.declare_parameter('fixed_transform_x',     0.0)
        self.declare_parameter('fixed_transform_y',     0.0)
        self.declare_parameter('fixed_transform_z',     0.0)
        self.declare_parameter('fixed_transform_roll',  0.0)
        self.declare_parameter('fixed_transform_pitch', 0.0)
        self.declare_parameter('fixed_transform_yaw',   0.0)

        lidar_topic         = self.get_parameter('lidar_topic').value
        objects_topic       = self.get_parameter('objects_topic').value
        output_topic        = self.get_parameter('output_topic').value
        marker_topic        = self.get_parameter('marker_topic').value
        self.padding        = self.get_parameter('padding').value
        self.max_object_age = self.get_parameter('max_object_age').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.objects_tf_frame = self.get_parameter('objects_tf_frame').value or ''

        fx = self.get_parameter('fixed_transform_x').value
        fy = self.get_parameter('fixed_transform_y').value
        fz = self.get_parameter('fixed_transform_z').value
        fr = self.get_parameter('fixed_transform_roll').value
        fp = self.get_parameter('fixed_transform_pitch').value
        fw = self.get_parameter('fixed_transform_yaw').value
        if any(v != 0.0 for v in (fx, fy, fz, fr, fp, fw)):
            self._fixed_R = _rpy_to_rot(fr, fp, fw)
            self._fixed_t = np.array([fx, fy, fz], dtype=np.float64)
        else:
            self._fixed_R = None
            self._fixed_t = None

        self._tf = TFCache()
        self._latest_objects: ObjectsStamped | None = None
        self._total_removed = 0
        self._total_scans   = 0

        self._pub_cloud  = self.create_publisher(PointCloud2,   output_topic, 10)
        self._pub_marker = self.create_publisher(MarkerArray,   marker_topic, 10)
        self.create_subscription(TFMessage,      '/tf_static',  self._on_tf_static,  _QOS_TF_STATIC)
        self.create_subscription(TFMessage,      '/tf',         self._on_tf,         _QOS_TF)
        self.create_subscription(ObjectsStamped, objects_topic, self._on_objects,    _QOS_SENSOR)
        self.create_subscription(PointCloud2,    lidar_topic,   self._on_lidar,      _QOS_SENSOR)

        self.get_logger().info(
            f'\nDynamicObjectFilter ready'
            f'\n  lidar   → {lidar_topic}'
            f'\n  objects → {objects_topic}'
            f'\n  output  → {output_topic}'
            f'\n  markers → {marker_topic}'
            f'\n  padding {self.padding} m | max_age {self.max_object_age} s | '
            f'min_conf {self.min_confidence}'
        )

    # ------------------------------------------------------------------

    def _on_tf_static(self, msg: TFMessage) -> None:
        self._tf.update(msg)
        self.get_logger().info(
            f'TF cache: {self._tf.known_frames}', throttle_duration_sec=5.0)

    def _on_tf(self, msg: TFMessage) -> None:
        self._tf.update(msg)

    def _on_objects(self, msg: ObjectsStamped) -> None:
        self._latest_objects = msg

    def _on_lidar(self, cloud: PointCloud2) -> None:
        self._total_scans += 1
        objects = self._latest_objects

        if objects is None or not objects.objects:
            self._pub_cloud.publish(cloud)
            return

        t_cloud = Time.from_msg(cloud.header.stamp).nanoseconds * 1e-9
        t_obj   = Time.from_msg(objects.header.stamp).nanoseconds * 1e-9
        if abs(t_cloud - t_obj) > self.max_object_age:
            self.get_logger().warn(
                f'Objects stale ({abs(t_cloud - t_obj):.2f} s)', throttle_duration_sec=2.0)
            self._pub_cloud.publish(cloud)
            return

        obbs, corners_lidar_list, labels = self._transform_objects(objects, cloud.header.frame_id)

        self._publish_markers(corners_lidar_list, labels, objects, cloud.header.frame_id)

        if not obbs:
            self._pub_cloud.publish(cloud)
            return

        self._pub_cloud.publish(self._filter(cloud, obbs))

    # ------------------------------------------------------------------

    def _transform_objects(self, objects: ObjectsStamped, lidar_frame: str):
        """
        Returns (obbs, corners_list, labels) for all objects above confidence threshold.
        obbs: list of (center, axes, half_extents) in LiDAR frame
        corners_list: matching list of (8,3) corner arrays in LiDAR frame
        labels: matching list of label strings
        """
        src_frame = self.objects_tf_frame or objects.header.frame_id
        if self._fixed_R is not None:
            R, tvec = self._fixed_R, self._fixed_t
        else:
            try:
                R, tvec = self._tf.lookup(lidar_frame, src_frame)
            except LookupError as exc:
                self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
                return [], [], []

        obbs, corners_list, labels = [], [], []
        for obj in objects.objects:
            if obj.confidence < self.min_confidence:
                continue
            corners_raw = np.array(
                [[c.kp[0], c.kp[1], c.kp[2]] for c in obj.bounding_box_3d.corners],
                dtype=np.float64,
            )
            if corners_raw.shape[0] != 8:
                continue

            corners_lidar = (R @ corners_raw.T).T + tvec
            obb = _build_obb(corners_lidar, self.padding)
            if obb is not None:
                obbs.append(obb)
                corners_list.append(corners_lidar)
                labels.append(obj.label)

        return obbs, corners_list, labels

    def _publish_markers(self, corners_list, labels, objects, lidar_frame):
        ma = MarkerArray()
        stamp = objects.header.stamp
        for i, (corners, label) in enumerate(zip(corners_list, labels)):
            center = corners.mean(axis=0)
            obj = objects.objects[i]
            ma.markers.append(_box_marker(i * 2,     corners, label, lidar_frame, stamp))
            ma.markers.append(_text_marker(i * 2 + 1, center, label, obj.confidence,
                                           lidar_frame, stamp))
        # Delete stale markers from previous frames
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        ma.markers.insert(0, delete_all)
        self._pub_marker.publish(ma)

    def _filter(self, cloud: PointCloud2, obbs: list) -> PointCloud2:
        n    = cloud.width * cloud.height
        step = cloud.point_step
        raw  = np.frombuffer(cloud.data, dtype=np.uint8).reshape(n, step)
        xyz  = _extract_xyz(cloud)

        remove = np.zeros(n, dtype=bool)
        for center, axes, half_extents in obbs:
            remove |= _points_in_obb(xyz, center, axes, half_extents)

        keep      = ~remove
        n_kept    = int(keep.sum())
        n_removed = n - n_kept
        self._total_removed += n_removed

        if self._total_scans % 50 == 0:
            self.get_logger().info(
                f'Scan #{self._total_scans}: removed {n_removed}/{n} pts '
                f'({len(obbs)} boxes) | total removed: {self._total_removed}'
            )

        out              = PointCloud2()
        out.header       = cloud.header
        out.fields       = cloud.fields
        out.is_bigendian = cloud.is_bigendian
        out.point_step   = step
        out.height       = 1
        out.width        = n_kept
        out.row_step     = n_kept * step
        out.is_dense     = False
        out.data         = raw[keep].tobytes()
        return out


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObjectFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
