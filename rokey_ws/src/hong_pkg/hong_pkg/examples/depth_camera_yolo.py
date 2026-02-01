#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

import numpy as np
import cv2
from cv_bridge import CvBridge

# ================================
# 설정 상수
# ================================
DEPTH_TOPIC = '/robot4/oakd/stereo/image_raw'        # Depth 이미지 토픽
CAMERA_INFO_TOPIC = '/robot4/oakd/stereo/camera_info'  # CameraInfo 토픽
CENTER_POS_TOPIC = '/center_pos'                       # (u, v) 픽셀 좌표 토픽

NORMALIZE_DEPTH_RANGE = 3.0    # 시각화 정규화 범위 (m)
# ================================


class DepthChecker(Node):
    def __init__(self):
        super().__init__('depth_checker')
        self.bridge = CvBridge()
        self.K = None
        self.should_exit = False

        # latest (u, v) from /center_pos (Float32MultiArray: [u, v, ...])
        self.latest_uv = None

        # Depth image subscription
        self.subscription = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10
        )

        # CameraInfo subscription
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10
        )

        # center_pos subscription (x, y in RGB/depth image pixel coordinates)
        self.center_pos_subscription = self.create_subscription(
            Float32MultiArray,
            CENTER_POS_TOPIC,
            self.center_pos_callback,
            10
        )

    # ---------------------------
    # Callbacks
    # ---------------------------

    def camera_info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo received: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, "
                f"cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}"
            )

    def center_pos_callback(self, msg: Float32MultiArray):
        """
        Expecting:
          msg.data[0] = u (x pixel)
          msg.data[1] = v (y pixel)
        in the same resolution / alignment as the depth image.
        """
        if len(msg.data) < 2:
            self.get_logger().warn(
                f"/center_pos received but data length < 2: {len(msg.data)}"
            )
            return

        u = float(msg.data[0])
        v = float(msg.data[1])
        self.latest_uv = (u, v)
        self.get_logger().debug(f"Updated center_pos: u={u:.2f}, v={v:.2f}")

    def depth_callback(self, msg: Image):
        if self.should_exit:
            return

        if self.K is None:
            self.get_logger().warn('Waiting for CameraInfo...')
            return

        # Convert depth image (uint16 or float32) – assuming mm
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        if depth_mm is None:
            self.get_logger().warn('Failed to convert depth image')
            return

        if depth_mm.ndim != 2:
            self.get_logger().warn(
                f"Depth image is not single-channel. shape={depth_mm.shape}"
            )
            return

        height, width = depth_mm.shape

        # ----------------------------
        # Choose pixel (u, v)
        # ----------------------------
        if self.latest_uv is not None:
            u_f, v_f = self.latest_uv
            u = int(round(u_f))
            v = int(round(v_f))
        else:
            # Fallback: use principal point (cx, cy) from intrinsics
            cx = self.K[0, 2]
            cy = self.K[1, 2]
            u = int(cx)
            v = int(cy)
            self.get_logger().warn(
                "No /center_pos received yet, using camera center (cx, cy)."
            )

        # Clamp to image bounds
        u = max(0, min(width - 1, u))
        v = max(0, min(height - 1, v))

        # ----------------------------
        # Read depth at (u, v)
        # ----------------------------
        distance_mm = depth_mm[v, u]

        # Handle float32 NaN
        if isinstance(distance_mm, float) and np.isnan(distance_mm):
            distance_m = float('nan')
        else:
            distance_m = float(distance_mm) / 1000.0  # mm → m

        self.get_logger().info(
            f"Image size: {width}x{height}, "
            f"Distance at (u={u}, v={v}) = {distance_m:.3f} meters"
        )

        # ----------------------------
        # Visualization
        # ----------------------------
        depth_vis = np.nan_to_num(depth_mm, nan=0.0)
        depth_vis = np.clip(depth_vis, 0, NORMALIZE_DEPTH_RANGE * 1000)  # mm
        depth_vis = (depth_vis / (NORMALIZE_DEPTH_RANGE * 1000) * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Mark the chosen pixel (u, v)
        cv2.circle(depth_colored, (u, v), 5, (0, 0, 0), -1)
        cv2.line(depth_colored, (0, v), (width, v), (0, 0, 0), 1)
        cv2.line(depth_colored, (u, 0), (u, height), (0, 0, 0), 1)

        cv2.imshow('Depth Image (u,v from /center_pos)', depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True

# ---------------------------
# main
# ---------------------------

def main():
    rclpy.init()
    node = DepthChecker()

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
