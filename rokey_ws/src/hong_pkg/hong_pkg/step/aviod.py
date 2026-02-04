#!/usr/bin/env python3
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from message_filters import Subscriber, ApproximateTimeSynchronizer

from ultralytics import YOLO


class YoloDepthToScan(Node):
    """
    ns = /robot5
    depth_topic = {ns}/oakd/stereo/image_raw
    rgb_topic   = {ns}/oakd/rgb/image_raw/compressed
    info_topic  = {ns}/oakd/rgb/camera_info

    RGB-Depth는 픽셀 정렬
    YOLO(best.pt)로 bbox -> Depth로 거리 -> TF(camera->base_link) -> LaserScan 생성
    """

    def __init__(self):
        super().__init__('yolo_depth_to_scan')

        # -----------------------------
        # Topics (namespace-aware)
        # -----------------------------
        ns = self.get_namespace()  # e.g. "/robot5"
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'
        self.rgb_topic   = f'{ns}/oakd/rgb/image_raw/compressed'
        self.info_topic  = f'{ns}/oakd/rgb/camera_info'

        self.scan_topic  = f'{ns}/yolo_scan'

        # -----------------------------
        # Params / Tunables
        # -----------------------------
        self.model_path = 'best.pt'     # TODO: 절대경로 추천
        self.conf_thres = 0.35
        self.max_det = 20

        # depth filtering
        self.z_min = 0.20
        self.z_max = 6.00

        # bbox ROI depth averaging (odd number recommended)
        self.roi_half = 2  # 2 => 5x5 평균

        # LaserScan config
        self.angle_min = -math.pi / 2
        self.angle_max =  math.pi / 2
        self.angle_inc = math.radians(1.0)   # 1 deg
        self.range_min = self.z_min
        self.range_max = self.z_max

        # costmap에 더 두껍게 찍고 싶으면 spread_deg 키우기
        self.spread_deg = 6.0

        # TF target frame (Nav2 robot_base_frame과 동일하게 맞추기)
        self.base_frame = f'{ns.strip("/")}/base_link'  # "robot5/base_link"
        # 만약 TF가 그냥 "base_link"면 위 줄 대신 아래로:
        # self.base_frame = 'base_link'

        # -----------------------------
        # State
        # -----------------------------
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.get_logger().info(f'Loaded YOLO model: {self.model_path}')

        self.K = None
        self.camera_frame = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.pub_scan = self.create_publisher(LaserScan, self.scan_topic, 10)

        # CameraInfo subscriber (단독)
        self.sub_info = self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 1)

        # Sync RGB + Depth
        self.sub_rgb = Subscriber(self, CompressedImage, self.rgb_topic)
        self.sub_depth = Subscriber(self, Image, self.depth_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.cb_sync)

        # Inference rate limit (optional)
        self.last_infer_t = 0.0
        self.min_infer_dt = 0.07  # ~14Hz

        self.get_logger().info(
            f'Subscribed:\n  RGB:   {self.rgb_topic}\n  Depth: {self.depth_topic}\n  Info:  {self.info_topic}\n'
            f'Publishing:\n  Scan:  {self.scan_topic}\n'
            f'Base frame: {self.base_frame}'
        )

    # -----------------------------
    # CameraInfo callback
    # -----------------------------
    def cb_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.camera_frame = msg.header.frame_id
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        self.get_logger().info(f'CameraInfo received: frame={self.camera_frame} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')

    # -----------------------------
    # Main sync callback
    # -----------------------------
    def cb_sync(self, rgb_msg: CompressedImage, depth_msg: Image):
        if self.K is None or self.camera_frame is None:
            return

        now = time.time()
        if now - self.last_infer_t < self.min_infer_dt:
            return
        self.last_infer_t = now

        # 1) Decode RGB compressed
        np_arr = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Failed to decode RGB compressed image')
            return

        h, w = frame.shape[:2]

        # 2) Depth image -> cv2
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth decode failed: {e}')
            return

        # depth encoding handling
        # - 16UC1: usually mm
        # - 32FC1: usually meters
        depth_is_mm = (depth.dtype == np.uint16)
        depth_is_m  = (depth.dtype == np.float32) or (depth.dtype == np.float64)

        if not (depth_is_mm or depth_is_m):
            self.get_logger().warn(f'Unsupported depth dtype: {depth.dtype}')
            return

        # 3) YOLO inference
        results = self.model.predict(frame, conf=self.conf_thres, max_det=self.max_det, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            self.publish_empty_scan()
            return

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        # 4) Prepare scan ranges
        n = int((self.angle_max - self.angle_min) / self.angle_inc) + 1
        ranges = [float('inf')] * n

        # 5) For each bbox: get center depth -> 3D point -> TF -> angle/range
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            # clamp bbox to image bounds
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            u = int(0.5 * (x1 + x2))
            v = int(0.5 * (y1 + y2))

            # ROI 평균 depth (노이즈/홀 줄이기)
            z = self.get_depth_roi_mean(depth, u, v, self.roi_half, depth_is_mm)
            if z is None:
                continue
            if not (self.z_min <= z <= self.z_max):
                continue

            # Camera frame 3D point (optical frame convention)
            # X right, Y down, Z forward (optical)
            Xc = (u - cx) * z / fx
            Yc = (v - cy) * z / fy
            Zc = z

            # Transform to base_link
            pt_cam = PointStamped()
            pt_cam.header.stamp = depth_msg.header.stamp  # sync time
            pt_cam.header.frame_id = self.camera_frame
            pt_cam.point.x = float(Xc)
            pt_cam.point.y = float(Yc)
            pt_cam.point.z = float(Zc)

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.camera_frame,
                    rclpy.time.Time()
                )
                pt_base = do_transform_point(pt_cam, tf)
            except Exception as e:
                # TF가 아직 준비 안됐을 수 있음
                self.get_logger().debug(f'TF transform failed: {e}')
                continue

            xb = pt_base.point.x
            yb = pt_base.point.y
            zb = pt_base.point.z

            # base_link 기준: x forward, y left (REP-103)
            # range는 xy 평면 거리로
            rng = math.sqrt(xb * xb + yb * yb)

            # 뒤쪽(x<=0)이면 회피대상에서 제외(원하면 제거)
            if xb <= 0.05:
                continue

            if not (self.range_min <= rng <= self.range_max):
                continue

            theta = math.atan2(yb, xb)
            if theta < self.angle_min or theta > self.angle_max:
                continue

            # spread 각도 만큼 두껍게 표시
            self.apply_scan_spread(ranges, theta, rng)

        # 6) publish scan
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self.base_frame
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges
        self.pub_scan.publish(scan)

    # -----------------------------
    # Helpers
    # -----------------------------
    def get_depth_roi_mean(self, depth, u, v, half, depth_is_mm: bool):
        h, w = depth.shape[:2]
        x0 = max(0, u - half)
        x1 = min(w - 1, u + half)
        y0 = max(0, v - half)
        y1 = min(h - 1, v + half)

        roi = depth[y0:y1 + 1, x0:x1 + 1]

        # 0이나 NaN 제거
        if depth_is_mm:
            vals = roi.astype(np.float32)
            vals = vals[vals > 0.0]  # 0mm 제거
            if vals.size == 0:
                return None
            z_m = float(np.median(vals)) * 0.001  # mm->m (median이 더 안정적)
            return z_m
        else:
            vals = roi.astype(np.float32)
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0.0]
            if vals.size == 0:
                return None
            z_m = float(np.median(vals))  # already meters
            return z_m

    def apply_scan_spread(self, ranges, theta, rng):
        spread = math.radians(self.spread_deg)
        a0 = theta - spread / 2
        a1 = theta + spread / 2

        n = len(ranges)
        i0 = int((a0 - self.angle_min) / self.angle_inc)
        i1 = int((a1 - self.angle_min) / self.angle_inc)
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))

        lo, hi = min(i0, i1), max(i0, i1)
        for i in range(lo, hi + 1):
            ranges[i] = min(ranges[i], rng)

    def publish_empty_scan(self):
        n = int((self.angle_max - self.angle_min) / self.angle_inc) + 1
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self.base_frame
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = [float('inf')] * n
        self.pub_scan.publish(scan)


def main():
    rclpy.init()
    node = YoloDepthToScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

