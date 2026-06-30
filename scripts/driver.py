"""GraspNet grasp-candidate node.

Subscribes to the EgoHOS segmentation mask, the aligned depth image, and the
matching CameraInfo (synchronized by header stamp), restricts the depth point
cloud to the object class (mask == 2), and runs graspnet-baseline to produce
scored grasp candidates, published as a PoseArray.

Runs inside its own venv (graspnet_venv), isolated from the main ROS workspace
and from EgoHOS's venv, talking to the rest of the graph only over DDS topics
-- same isolation pattern as EgoHOS/scripts/driver.py.
"""

import sys
import time
from pathlib import Path

import cv2
import cv_bridge
import message_filters
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

# scripts/driver.py -> graspnet-baseline, regardless of where the repo is checked out
GRASPNET_ROOT = Path(__file__).resolve().parent.parent
for _sub in ('models', 'dataset', 'utils'):
    sys.path.append(str(GRASPNET_ROOT / _sub))

from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI import GraspGroup  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo as PointCloudCamera, create_point_cloud_from_depth_image  # noqa: E402

CHECKPOINT_DEFAULT = str(GRASPNET_ROOT / 'logs/log_rs/checkpoint.tar')

# EgoHOS combined mask classes (EgoHOS/scripts/driver.py): 0 background, 1 hand, 2 object
OBJECT_CLASS_ID = 2

STATUS_NO_HAND = 'no_hand'
STATUS_UNSTABLE = 'hand_unstable'
STATUS_STABLE = 'hand_stable'


class GraspNetNode(Node):

    def __init__(self):
        super().__init__('graspnet_node')

        self.declare_parameter('mask_topic', 'segmentation_mask')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('grasp_topic', 'grasp_candidates')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('debug_image_topic', 'grasp_debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_approach_length', 0.04)
        self.declare_parameter('hand_status_topic', 'hand_stability_status')
        self.declare_parameter('checkpoint_path', CHECKPOINT_DEFAULT)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('max_grasps', 50)
        self.declare_parameter('min_score', 0.2)
        # realsense aligned-depth is 16UC1 millimetres; this converts raw depth to metres
        self.declare_parameter('depth_scale', 1000.0)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('save_debug_inputs_dir', '/tmp/graspnet_inputs')

        device_name = self.get_parameter('device').value
        self._device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self._num_point = self.get_parameter('num_point').value
        self._voxel_size = self.get_parameter('voxel_size').value
        self._collision_thresh = self.get_parameter('collision_thresh').value
        self._max_grasps = self.get_parameter('max_grasps').value
        self._min_score = self.get_parameter('min_score').value
        self._depth_scale = self.get_parameter('depth_scale').value
        self._publish_debug_image = self.get_parameter('publish_debug_image').value
        self._debug_approach_length = self.get_parameter('debug_approach_length').value
        self._last_color_bgr = None
        self._last_hand_status = None
        self._trigger_pending = False

        self.get_logger().info(f'Loading GraspNet on {self._device}...')
        self._net = GraspNet(
            input_feature_dim=0, num_view=self.get_parameter('num_view').value, num_angle=12,
            num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False)
        self._net.to(self._device)
        checkpoint = torch.load(self.get_parameter('checkpoint_path').value, map_location=self._device)
        self._net.load_state_dict(checkpoint['model_state_dict'])
        self._net.eval()
        self.get_logger().info(f"GraspNet loaded (epoch {checkpoint['epoch']}).")

        self._bridge = cv_bridge.CvBridge()
        self._busy = False

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        mask_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('mask_topic').value, qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter('depth_topic').value, qos_profile=sensor_qos)
        info_sub = message_filters.Subscriber(
            self, CameraInfo, self.get_parameter('camera_info_topic').value, qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [mask_sub, depth_sub, info_sub], queue_size=30,
            slop=self.get_parameter('sync_slop').value)
        self._sync.registerCallback(self._on_synced)

        self.create_subscription(
            String, self.get_parameter('hand_status_topic').value, self._on_hand_status, 10)

        self._grasp_pub = self.create_publisher(PoseArray, self.get_parameter('grasp_topic').value, 1)

        if self._publish_debug_image:
            self.create_subscription(
                Image, self.get_parameter('color_topic').value, self._on_color, sensor_qos)
            self._debug_pub = self.create_publisher(
                Image, self.get_parameter('debug_image_topic').value, 1)

    def _on_color(self, msg: Image):
        self._last_color_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_hand_status(self, msg: String):
        if msg.data == STATUS_STABLE and self._last_hand_status != STATUS_STABLE:
            self._trigger_pending = True
        self._last_hand_status = msg.data

    def _on_synced(self, mask_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self._busy or not self._trigger_pending:
            return
        self._busy = True
        self._trigger_pending = False
        try:
            success = self._process(mask_msg, depth_msg, info_msg)
            # If inference found no valid grasps (e.g. bad depth, collisions, or missing segmentation)
            # but the hand is still stable, queue another attempt.
            if not success and self._last_hand_status == STATUS_STABLE:
                self._trigger_pending = True
        finally:
            self._busy = False

    def _process(self, mask_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        t0 = time.monotonic()
        mask = self._bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        camera = PointCloudCamera(
            info_msg.width, info_msg.height,
            info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5],
            self._depth_scale)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        valid = (mask == OBJECT_CLASS_ID) & (depth > 0)
        cloud_masked = cloud[valid]
        if len(cloud_masked) < 100:
            self.get_logger().warn(
                'Too few object points for grasp inference, skipping frame.', throttle_duration_sec=5.0)
            return False

        if len(cloud_masked) >= self._num_point:
            idxs = np.random.choice(len(cloud_masked), self._num_point, replace=False)
        else:
            idxs = np.concatenate([
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), self._num_point - len(cloud_masked), replace=True),
            ])
        cloud_sampled = torch.from_numpy(
            cloud_masked[idxs][np.newaxis].astype(np.float32)).to(self._device)

        with torch.no_grad():
            end_points = self._net({'point_clouds': cloud_sampled})
            grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

        if self._collision_thresh > 0:
            detector = ModelFreeCollisionDetector(cloud_masked, voxel_size=self._voxel_size)
            collision_mask = detector.detect(
                gg, approach_dist=0.05, collision_thresh=self._collision_thresh)
            gg = gg[~collision_mask]

        gg.nms()
        if len(gg) > 0:
            top_score = gg.scores.max()
            self.get_logger().info(f'Max grasp score before filtering: {top_score:.3f}', throttle_duration_sec=2.0)
        gg = gg[gg.scores >= self._min_score]
        gg.sort_by_score()  # high to low
        gg = gg[:self._max_grasps]

        out_dir = self.get_parameter('save_debug_inputs_dir').value
        if out_dir or self._publish_debug_image:
            canvas = self._create_debug_canvas(gg, camera, mask, depth)

            if self._publish_debug_image:
                self._publish_debug(canvas, mask_msg.header)

            if out_dir:
                import os
                os.makedirs(out_dir, exist_ok=True)
                timestamp = f"{mask_msg.header.stamp.sec}_{mask_msg.header.stamp.nanosec}"
                mask_vis = (mask * 127).astype(np.uint8)
                depth_vis = self._colorize_depth(depth)
                if self._last_color_bgr is not None and self._last_color_bgr.shape[:2] == mask.shape:
                    color_vis = self._last_color_bgr
                else:
                    color_vis = np.zeros_like(depth_vis)
                concat = np.hstack([color_vis, cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR), depth_vis, canvas])
                filepath = os.path.join(out_dir, f"inputs_and_outputs_{timestamp}.jpg")
                cv2.imwrite(filepath, concat)
                self.get_logger().info(f"Saved debug images to {filepath}")

        pose_array = PoseArray()
        pose_array.header = mask_msg.header
        if len(gg) > 0:
            # GraspNet gives X as approach, Y as binormal.
            # Franka Hand expects Z as approach, Y as binormal, and -X as orthogonal.
            # We post-multiply to map the axes to the Franka Hand coordinate frame.
            T_align = np.array([
                [ 0.0,  0.0,  1.0],
                [ 0.0,  1.0,  0.0],
                [-1.0,  0.0,  0.0]
            ])
            aligned_matrices = gg.rotation_matrices @ T_align
            quats = Rotation.from_matrix(aligned_matrices).as_quat()
            for translation, quat in zip(gg.translations, quats):
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = translation.tolist()
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat.tolist()
                pose_array.poses.append(pose)
        self._grasp_pub.publish(pose_array)

        self.get_logger().info(
            f'GraspNet inference took {time.monotonic() - t0:.3f}s, {len(pose_array.poses)} grasps',
            throttle_duration_sec=5.0)
            
        return len(pose_array.poses) > 0

    def _create_debug_canvas(self, gg, camera, mask, depth):
        if self._last_color_bgr is not None and self._last_color_bgr.shape[:2] == mask.shape:
            canvas = self._last_color_bgr.copy()
        else:
            canvas = self._colorize_depth(depth)

        object_px = mask == OBJECT_CLASS_ID
        canvas[object_px] = (canvas[object_px] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)

        self._draw_grasps(canvas, gg, camera)
        return canvas

    def _publish_debug(self, canvas, header):
        debug_msg = self._bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    @staticmethod
    def _colorize_depth(depth):
        valid = depth > 0
        norm = np.zeros(depth.shape, dtype=np.uint8)
        if valid.any():
            lo, hi = depth[valid].min(), depth[valid].max()
            norm[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _draw_grasps(self, canvas, gg, camera):
        if len(gg) == 0:
            return
        scores = gg.scores
        score_span = max(scores.max() - scores.min(), 1e-6)
        for i in range(len(gg)):
            center = gg.translations[i]
            approach = gg.rotation_matrices[i][:, 0]  # gripper approach axis
            binormal = gg.rotation_matrices[i][:, 1]  # finger separation axis
            half_width = gg.widths[i] / 2.0
            points_3d = np.stack([
                center,
                center + approach * self._debug_approach_length,
                center - binormal * half_width,
                center + binormal * half_width,
            ])
            if (points_3d[:, 2] <= 0).any():
                continue
            u = camera.fx * points_3d[:, 0] / points_3d[:, 2] + camera.cx
            v = camera.fy * points_3d[:, 1] / points_3d[:, 2] + camera.cy
            base, tip, left, right = (
                tuple(map(int, p)) for p in np.stack([u, v], axis=1).round())

            color = self._score_color((scores[i] - scores.min()) / score_span)
            cv2.line(canvas, left, right, color, 2)
            cv2.arrowedLine(canvas, tip, base, color, 2, tipLength=0.3)

    @staticmethod
    def _score_color(score_norm):
        # BGR, red (low score) -> green (high score)
        return (0, int(255 * score_norm), int(255 * (1 - score_norm)))


def main(args=None):
    rclpy.init(args=args)
    node = GraspNetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
