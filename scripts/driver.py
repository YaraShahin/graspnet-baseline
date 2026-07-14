"""GraspNet grasp-candidate node."""

import sys
import time
from pathlib import Path

import cv2
import cv_bridge
import message_filters
import numpy as np
import rclpy
import torch
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PointStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty

# Default path to the graspnet-baseline library
GRASPNET_ROOT = Path(__file__).resolve().parent.parent
for _sub in ('models', 'dataset', 'utils'):
    sys.path.append(str(GRASPNET_ROOT / _sub))

from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI import GraspGroup  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo as PointCloudCamera, create_point_cloud_from_depth_image  # noqa: E402

CHECKPOINT_DEFAULT = str(GRASPNET_ROOT / 'logs/log_rs/checkpoint.tar')

# Mask classes definition
HAND_CLASS_ID = 1
OBJECT_CLASS_ID = 2


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
        
        # Topic for the final selected grasp to render debug overlays
        self.declare_parameter('selected_grasp_topic', 'selected_grasp')
        self.declare_parameter('selected_debug_image_topic', 'selected_grasp_debug_image')
        self.declare_parameter('debug_approach_length', 0.04)
        
        # Trigger topic for one-shot inference, with retry attempts on failure
        self.declare_parameter('capture_trigger_topic', 'capture_trigger')
        self.declare_parameter('capture_attempts', 10)
        self.declare_parameter('checkpoint_path', CHECKPOINT_DEFAULT)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('voxel_size', 0.01)
        self.declare_parameter('collision_thresh', 0.01)
        self.declare_parameter('max_grasps', 50)
        self.declare_parameter('min_score', 0.25)
        
        # Radius (px) to exclude object points near the hand from grasp seeding
        self.declare_parameter('hand_exclusion_px', 20)
        
        # Depth scale factor to convert raw depth to meters
        self.declare_parameter('depth_scale', 1000.0)
        # Depth outlier filter: remove object points whose depth deviates by
        # more than N sigma-equivalents (MAD-based) from the median.
        # Set to 0 to disable.
        self.declare_parameter('depth_outlier_sigma', 2.5)
        self.declare_parameter('sync_slop', 0.05)
        
        # Buffer size to handle EgoHOS latency and synchronize matching frames
        self.declare_parameter('sync_queue_size', 150)
        self.declare_parameter('save_debug_inputs_dir', '/tmp/graspnet_inputs')

        device_name = self.get_parameter('device').value
        self._device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self._num_point = self.get_parameter('num_point').value
        self._voxel_size = self.get_parameter('voxel_size').value
        self._collision_thresh = self.get_parameter('collision_thresh').value
        self._max_grasps = self.get_parameter('max_grasps').value
        self._min_score = self.get_parameter('min_score').value
        exclusion_px = self.get_parameter('hand_exclusion_px').value
        self._hand_exclusion_kernel = None
        if exclusion_px > 0:
            self._hand_exclusion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * exclusion_px + 1, 2 * exclusion_px + 1))
        self._depth_scale = self.get_parameter('depth_scale').value
        self._depth_outlier_sigma = self.get_parameter('depth_outlier_sigma').value
        self._publish_debug_image = self.get_parameter('publish_debug_image').value
        self._debug_approach_length = self.get_parameter('debug_approach_length').value
        self._last_color_bgr = None
        self._trigger_pending = False
        self._attempts_left = 0
        self._capture_attempts = self.get_parameter('capture_attempts').value

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
        
        # Busy flag is used to drop frames if the pipeline is still processing the previous one
        self._busy = False
        
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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
            [mask_sub, depth_sub, info_sub],
            queue_size=self.get_parameter('sync_queue_size').value,
            slop=self.get_parameter('sync_slop').value)
        self._sync.registerCallback(self._on_synced)

        self.create_subscription(
            Empty, self.get_parameter('capture_trigger_topic').value, self._on_capture_trigger, 10)

        self._grasp_pub = self.create_publisher(PoseArray, self.get_parameter('grasp_topic').value, 1)
        
        # Hand centroid topic, used by selection node to score ergonomic clearance
        self.declare_parameter('hand_center_topic', 'hand_center')
        self._hand_center_pub = self.create_publisher(
            PointStamped, self.get_parameter('hand_center_topic').value, 1)

        self._last_inference = None
        self.create_subscription(
            PoseStamped, self.get_parameter('selected_grasp_topic').value,
            self._on_selected_grasp, 10)
        self._selected_debug_pub = self.create_publisher(
            Image, self.get_parameter('selected_debug_image_topic').value, 1)

        if self._publish_debug_image:
            self.create_subscription(
                Image, self.get_parameter('color_topic').value, self._on_color, sensor_qos)
            self._debug_pub = self.create_publisher(
                Image, self.get_parameter('debug_image_topic').value, 1)

    def _on_color(self, msg: Image):
        self._last_color_bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_capture_trigger(self, _msg: Empty):
        self.get_logger().info('Capture trigger received.')
        self._trigger_pending = True
        self._attempts_left = self._capture_attempts

    def _on_synced(self, mask_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self._busy or not self._trigger_pending:
            return
        self._busy = True
        self._trigger_pending = False
        try:
            success = self._process(mask_msg, depth_msg, info_msg)
            # Retry on the next synced frame if no valid grasps are found
            if not success:
                self._attempts_left -= 1
                if self._attempts_left > 0:
                    self._trigger_pending = True
                else:
                    self.get_logger().warn(
                        'No valid grasps after all capture attempts — giving up until next trigger.')
        finally:
            self._busy = False

    def _process(self, mask_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        t0 = time.monotonic()
        mask = self._bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        # Extract camera intrinsics (fx, fy, cx, cy) from camera_info message array
        camera = PointCloudCamera(
            info_msg.width, info_msg.height,
            info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5],
            self._depth_scale)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        object_mask = mask == OBJECT_CLASS_ID
        if self._hand_exclusion_kernel is not None:
            hand_mask = (mask == HAND_CLASS_ID).astype(np.uint8)
            if hand_mask.any():
                object_mask &= cv2.dilate(hand_mask, self._hand_exclusion_kernel) == 0

        valid = object_mask & (depth > 0)
        cloud_masked = cloud[valid]

        # Remove depth outlier points (multipath, edge bleed, temporal spikes)
        # using MAD-based filtering on the camera-frame Z axis.
        if self._depth_outlier_sigma > 0 and len(cloud_masked) > 10:
            z_vals = cloud_masked[:, 2]
            z_median = np.median(z_vals)
            mad = np.median(np.abs(z_vals - z_median))
            # 1.4826 converts MAD to standard-deviation equivalent for normal data
            sigma_equiv = max(1.4826 * mad, 0.005)  # 5 mm floor for thin objects
            z_lo = z_median - self._depth_outlier_sigma * sigma_equiv
            z_hi = z_median + self._depth_outlier_sigma * sigma_equiv
            inlier = (z_vals >= z_lo) & (z_vals <= z_hi)
            n_outliers = int((~inlier).sum())
            if n_outliers > 0:
                self.get_logger().info(
                    f'Depth outlier filter: removed {n_outliers}/{len(z_vals)} '
                    f'points outside [{z_lo:.3f}, {z_hi:.3f}] m '
                    f'(median {z_median:.3f}, MAD {mad:.4f})')
            cloud_masked = cloud_masked[inlier]

        if len(cloud_masked) < 100:
            self.get_logger().warn(
                'Too few object points for grasp inference, skipping frame.', throttle_duration_sec=5.0)
            return False

        # GraspNet requires exactly 'num_point' input points. 
        # Randomly downsample if we have too many, or oversample (with replacement) if too few.
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

        n_decoded = len(gg)
        if self._collision_thresh > 0:
            # Collision-check against both hand and object to avoid human contact
            collision_cloud = cloud[(mask > 0) & (depth > 0)]
            detector = ModelFreeCollisionDetector(collision_cloud, voxel_size=self._voxel_size)
            collision_mask = detector.detect(
                gg, approach_dist=0.05, collision_thresh=self._collision_thresh)
            gg = gg[~collision_mask]
        n_collision_free = len(gg)

        gg.nms()
        top_score = gg.scores.max() if len(gg) > 0 else float('nan')
        gg = gg[gg.scores >= self._min_score]

        # Log grasp survival funnel through collision checking and NMS
        self.get_logger().info(
            f'Grasps: {n_decoded} decoded -> {n_collision_free} collision-free -> '
            f'top score {top_score:.2f} after NMS -> {len(gg)} above min_score={self._min_score}, '
            f'object points {len(cloud_masked)}')

        gg.sort_by_score()
        gg = gg[:self._max_grasps]

        out_dir = self.get_parameter('save_debug_inputs_dir').value
        if out_dir or self._publish_debug_image:
            canvas = self._create_debug_canvas(gg[0:50], camera, mask, depth)

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

        # Compute and publish hand centroid if visible
        hand_valid = (mask == HAND_CLASS_ID) & (depth > 0)
        if hand_valid.any():
            hand_center = cloud[hand_valid].mean(axis=0)
            hand_msg = PointStamped()
            hand_msg.header = mask_msg.header
            hand_msg.point.x, hand_msg.point.y, hand_msg.point.z = hand_center.tolist()
            self._hand_center_pub.publish(hand_msg)

        pose_array = PoseArray()
        pose_array.header = mask_msg.header
        # EgoHOS segmentation returns color optical frame. The 180 Z-flip is handled 
        # statically in the TF tree via camera_color_optical_corrected.
        pose_array.header.frame_id = 'camera_color_optical_corrected'
        if len(gg) > 0:
            # Map GraspNet axes (X: approach, Y: binormal) to Franka Hand frame
            # (Z: approach, Y: binormal, -X: orthogonal)
            T_align = np.array([
                [ 0.0,  0.0,  1.0],
                [ 0.0,  1.0,  0.0],
                [-1.0,  0.0,  0.0]
            ])
            aligned_matrices = gg.rotation_matrices @ T_align
            quats = Rotation.from_matrix(aligned_matrices).as_quat()
            
            new_translations = gg.translations
            
            for translation, quat in zip(new_translations, quats):
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = translation.tolist()
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat.tolist()
                pose_array.poses.append(pose)
        self._grasp_pub.publish(pose_array)
        
        if len(gg) > 0:
            # Cache inference state to render final selected grasp debug image
            self._last_inference = (gg, camera, mask, depth, mask_msg.header)

        self.get_logger().info(
            f'GraspNet inference took {time.monotonic() - t0:.3f}s, {len(pose_array.poses)} grasps',
            throttle_duration_sec=5.0)

        return len(pose_array.poses) > 0

    def _on_selected_grasp(self, msg: PoseStamped):
        """Match selected grasp pose to cached inference candidates and render debug image."""
        if self._last_inference is None:
            return
        gg, camera, mask, depth, header = self._last_inference
        p = msg.pose.position
        dists = np.linalg.norm(
            gg.translations - np.array([p.x, p.y, p.z]), axis=1)
        idx = int(dists.argmin())
        if dists[idx] > 0.005:
            self.get_logger().warn(
                f'Selected grasp is {dists[idx]:.3f} m from the nearest candidate '
                'of the last inference — rendering the nearest one anyway.')
        self.get_logger().info(
            f'Selected grasp: candidate index {idx} of {len(gg)}, '
            f'GraspNet confidence score {gg.scores[idx]:.3f} '
            f'(top score this inference: {gg.scores.max():.3f}).')
        canvas = self._create_debug_canvas(gg[idx:idx + 1], camera, mask, depth)
        img_msg = self._bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
        img_msg.header = header
        self._selected_debug_pub.publish(img_msg)

        out_dir = self.get_parameter('save_debug_inputs_dir').value
        if out_dir:
            import os
            os.makedirs(out_dir, exist_ok=True)
            timestamp = f"{header.stamp.sec}_{header.stamp.nanosec}"
            filepath = os.path.join(out_dir, f'final_grasp_{timestamp}.jpg')
            cv2.imwrite(filepath, canvas)
            self.get_logger().info(f'Saved selected-grasp image to {filepath}')

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
            
            # Define grasp geometry keypoints in 3D: base, tip, left finger, right finger
            points_3d = np.stack([
                center,
                center + approach * self._debug_approach_length,
                center - binormal * half_width,
                center + binormal * half_width,
            ])
            
            # Skip drawing if any point is behind the camera plane (Z <= 0)
            if (points_3d[:, 2] <= 0).any():
                continue
                
            # Pinhole camera projection: map 3D points (X,Y,Z) to 2D image plane (u,v)
            u = camera.fx * points_3d[:, 0] / points_3d[:, 2] + camera.cx
            v = camera.fy * points_3d[:, 1] / points_3d[:, 2] + camera.cy
            base, tip, left, right = (
                tuple(map(int, p)) for p in np.stack([u, v], axis=1).round())

            color = self._score_color((scores[i] - scores.min()) / score_span)
            cv2.line(canvas, left, right, color, 2)
            cv2.arrowedLine(canvas, tip, base, color, 2, tipLength=0.3)

    @staticmethod
    def _score_color(score_norm):
        # BGR, red (low score), green (high score)
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
