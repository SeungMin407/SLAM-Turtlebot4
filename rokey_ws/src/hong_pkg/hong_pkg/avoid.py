#!/usr/bin/env python3
import math
import time
import numpy as np
import cv2
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

from ultralytics import YOLO


class YoloDepthToScan(Node):
    """
    최신 프레임만 처리(큐 누적 지연 제거)

    ⭐ PATCH(원형 충돌범위 느낌):
    - LaserScan은 1각도당 range 1개라서 "원형"을 직접 못 만듦.
    - 그래서 base_link 좌표계에서 장애물 중심(xb,yb) 기준으로
      반경 yolo_radius_m 원의 둘레를 여러 점 샘플링 -> 각 점을 (theta, range)로 변환해 ranges에 찍는다.
    - 결과적으로 costmap에서 디스크(원형) 장애물처럼 훨씬 잘 막힘.
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
        self.model_path = '/home/rokey/Desktop/project/gotoend_turtle/rokey_ws/src/hong_pkg/hong_pkg/yolov8s.pt'
        self.conf_thres = 0.35
        self.max_det = 20
        self.imgsz = 416

        # scan용 depth 범위
        self.z_min = 0.20
        self.z_max = 6.00

        # ROI depth median (half=2 => 5x5)
        self.roi_half = 2

        # LaserScan config (전방 180도)
        self.angle_min = -math.pi / 2
        self.angle_max =  math.pi / 2
        self.angle_inc = math.radians(1.0)   # 1 deg
        self.range_min = self.z_min
        self.range_max = self.z_max

        # -----------------------------
        # ⭐ 원형 반경(미터) 기반 "디스크 느낌" 확장
        # -----------------------------
        # 추천: 0.20~0.50부터 시작해서 튜닝
        self.yolo_radius_m = 0.30

        # 원 둘레 샘플 개수(많을수록 더 원형, but 계산량 증가)
        self.circle_samples = 16

        # 너무 로봇 바로 앞에서 과보수적으로 막히는 걸 방지할 최소거리
        self.min_x_front = 0.05

        # TF target frame
        self.base_frame = 'base_link'

        # -----------------------------
        # Debug options
        # -----------------------------
        self.show_debug = True
        self.win_name = 'YOLO center debug'
        self.show_dt = 0.10
        self._last_show_t = 0.0

        self.log_every_sec = 1.0
        self._last_log_t = 0.0
        self._last_tf_warn_t = 0.0

        self.debug_show_max_dist_m = None
        self.debug_scan_max_dist_m = None

        # YOLO 처리 주기
        self.min_infer_dt = 0.12     # ~8Hz
        self.last_infer_t = 0.0

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

        # Latest message buffers
        self.lock = Lock()
        self.last_rgb = None
        self.last_depth = None

        # Subscribers
        self.create_subscription(CameraInfo, self.info_topic, self.cb_info, 1)
        self.create_subscription(CompressedImage, self.rgb_topic, self.cb_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_profile_sensor_data)

        # Timer
        self.timer = self.create_timer(0.05, self.process_latest)

        self.get_logger().info(
            f'Subscribed:\n  RGB:   {self.rgb_topic}\n  Depth: {self.depth_topic}\n  Info:  {self.info_topic}\n'
            f'Publishing:\n  Scan:  {self.scan_topic}\n'
            f'Base frame: {self.base_frame}\n'
            f'YOLO radius (m): {self.yolo_radius_m}, circle_samples={self.circle_samples}'
        )

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

    def cb_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        self.camera_frame = msg.header.frame_id

        now = time.time()
        if now - self._last_log_t >= self.log_every_sec:
            self.get_logger().info(f"[CameraInfo] frame={self.camera_frame}")

    def cb_rgb(self, msg: CompressedImage):
        with self.lock:
            self.last_rgb = msg

    def cb_depth(self, msg: Image):
        with self.lock:
            self.last_depth = msg

    def process_latest(self):
        if self.K is None or self.camera_frame is None:
            return

        with self.lock:
            rgb_msg = self.last_rgb
            depth_msg = self.last_depth

        if rgb_msg is None or depth_msg is None:
            return

        now = time.time()
        if now - self.last_infer_t < self.min_infer_dt:
            return
        self.last_infer_t = now

        # Decode RGB
        np_arr = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        h, w = frame.shape[:2]

        # Depth -> cv2
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception:
            return

        depth_is_mm = (depth.dtype == np.uint16)
        depth_is_m  = (depth.dtype == np.float32) or (depth.dtype == np.float64)
        if not (depth_is_mm or depth_is_m):
            return

        # YOLO inference
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_thres,
                                     max_det=self.max_det, verbose=False)
        boxes = results[0].boxes if (results and len(results) > 0 and results[0].boxes is not None) else None
        det_count = int(len(boxes)) if boxes is not None else 0

        do_show = self.show_debug and (now - self._last_show_t >= self.show_dt)
        overlay = frame.copy() if do_show else None
        if do_show:
            self._last_show_t = now

        # Prepare scan ranges
        n = int((self.angle_max - self.angle_min) / self.angle_inc) + 1
        ranges = [float('inf')] * n

        kept = 0
        dropped_nod = 0

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])

        if det_count == 0:
            self.publish_empty_scan()
            if do_show:
                cv2.putText(overlay, "det=0", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.imshow(self.win_name, overlay)
                cv2.waitKey(1)
            return

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))

            u = int(0.5 * (x1 + x2))
            v = int(0.5 * (y1 + y2))

            z = self.get_depth_roi_median(depth, u, v, self.roi_half, depth_is_mm)

            # debug marker
            if do_show:
                color = (0, 255, 255) if z is not None else (0, 0, 255)
                cv2.drawMarker(overlay, (u, v), color,
                               markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                if z is not None:
                    cv2.putText(overlay, f"{z:.2f}m", (u + 6, v - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(overlay, "no_depth", (u + 6, v - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if z is None:
                dropped_nod += 1
                continue

            if (self.debug_scan_max_dist_m is not None) and (z > self.debug_scan_max_dist_m):
                continue

            if not (self.z_min <= z <= self.z_max):
                continue

            # camera optical -> 3D
            Xc = (u - cx) * z / fx
            Yc = (v - cy) * z / fy
            Zc = z

            pt_cam = PointStamped()
            pt_cam.header.stamp = depth_msg.header.stamp
            pt_cam.header.frame_id = self.camera_frame
            pt_cam.point.x = float(Xc)
            pt_cam.point.y = float(Yc)
            pt_cam.point.z = float(Zc)

            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
                pt_base = do_transform_point(pt_cam, tf)
            except Exception as e:
                if now - self._last_tf_warn_t >= 1.0:
                    self._last_tf_warn_t = now
                    self.get_logger().warn(f"[TF FAIL] base={self.base_frame} cam={self.camera_frame} err={e}")
                continue

            xb = float(pt_base.point.x)
            yb = float(pt_base.point.y)

            # 전방 최소거리
            if xb <= self.min_x_front:
                continue

            # ⭐ 핵심: "중심점 + 원 둘레 샘플링"을 LaserScan에 찍기
            added = self.apply_disc_like_block(ranges, xb, yb, self.yolo_radius_m, self.circle_samples)
            if added > 0:
                kept += 1

        if do_show:
            cv2.putText(
                overlay,
                f"det={det_count} kept={kept} nod={dropped_nod} | R={self.yolo_radius_m:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2
            )
            cv2.imshow(self.win_name, overlay)
            cv2.waitKey(1)

        if now - self._last_log_t >= self.log_every_sec:
            self._last_log_t = now
            valid = [(i, r) for i, r in enumerate(ranges) if math.isfinite(r)]
            self.get_logger().info(
                f"[YOLO] det={det_count} kept={kept} nod={dropped_nod} | scan_valid={len(valid)} | R={self.yolo_radius_m:.2f}m"
            )

        # publish scan
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "base_link"   # 네 costmap robot_base_frame과 동일
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges
        self.pub_scan.publish(scan)

    def get_depth_roi_median(self, depth, u, v, half, depth_is_mm: bool):
        h, w = depth.shape[:2]
        x0 = max(0, u - half)
        x1 = min(w - 1, u + half)
        y0 = max(0, v - half)
        y1 = min(h - 1, v + half)

        roi = depth[y0:y1 + 1, x0:x1 + 1]

        if depth_is_mm:
            vals = roi.astype(np.float32)
            vals = vals[vals > 0.0]
            if vals.size == 0:
                return None
            return float(np.median(vals)) * 0.001
        else:
            vals = roi.astype(np.float32)
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0.0]
            if vals.size == 0:
                return None
            return float(np.median(vals))

    def apply_disc_like_block(self, ranges, xb, yb, R, samples):
        """
        base_link 좌표계에서 (xb, yb)를 중심으로 반경 R 원의 둘레를 samples개 찍고,
        각 점을 LaserScan의 (theta, range)로 변환해서 ranges에 "여러 점"을 넣는다.
        => costmap에서 거의 원형 장애물처럼 작동.
        """
        count = 0

        # 중심점도 포함 (가운데)
        count += self.apply_single_point_to_scan(ranges, xb, yb)

        # 원 둘레 샘플
        for k in range(samples):
            ang = 2.0 * math.pi * (k / float(samples))
            px = xb + R * math.cos(ang)
            py = yb + R * math.sin(ang)
            # 로봇 뒤쪽/너무 가까운 건 제외(과보수 방지)
            if px <= self.min_x_front:
                continue
            count += self.apply_single_point_to_scan(ranges, px, py)

        return count

    def apply_single_point_to_scan(self, ranges, x, y):
        theta = math.atan2(y, x)
        if theta < self.angle_min or theta > self.angle_max:
            return 0

        rng = math.sqrt(x * x + y * y)
        if not (self.range_min <= rng <= self.range_max):
            return 0

        i = int((theta - self.angle_min) / self.angle_inc)
        i = max(0, min(len(ranges) - 1, i))
        ranges[i] = min(ranges[i], rng)
        return 1

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
