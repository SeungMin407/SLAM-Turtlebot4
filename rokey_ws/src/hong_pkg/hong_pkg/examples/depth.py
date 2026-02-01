#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurtleBot4 OAK-D RGB + YOLO + Depth

- /robot4/oakd/rgb/preview/image_raw  에서 RGB 이미지 구독
- /robot4/oakd/stereo/image_raw       에서 Depth 이미지 구독
- YOLO로 바운딩 박스 검출
- 각 박스의 중심 (cx, cy) 위치에서 depth 픽셀값 읽어서 거리(m) 계산 후 화면에 표시

필요 패키지:
  pip install ultralytics opencv-python
  sudo apt install ros-humble-cv-bridge ros-humble-vision-opencv
"""

import os
import sys
import math
import threading
from queue import Queue
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import cv2
import numpy as np


# =========================
# 설정 (토픽 / 파라미터)
# =========================

# TurtleBot4 OAK-D RGB / Depth 토픽 (필요하면 바꿔서 사용)
RGB_TOPIC = '/robot4/oakd/rgb/preview/image_raw'
DEPTH_TOPIC = '/robot4/oakd/stereo/image_raw'   # 16UC1 (mm) 가정


class YOLODepthSubscriber(Node):
    def __init__(self, model):
        super().__init__('yolo_depth_subscriber')

        self.model = model
        self.bridge = CvBridge()

        # YOLO 입력용 이미지 큐 (최근 1장만)
        self.image_queue = Queue(maxsize=1)

        # 종료 플래그
        self.should_shutdown = False

        # 클래스 이름 (YOLO 모델에서 가져옴)
        self.classNames = model.names if hasattr(model, 'names') else ['Object']

        # 최신 depth 이미지
        self.latest_depth = None  # np.ndarray (H, W), mm 단위 가정

        # ========== ROS2 구독 설정 ==========

        # RGB 이미지 구독
        self.rgb_sub = self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            10
        )

        # Depth 이미지 구독
        self.depth_sub = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10
        )

        # YOLO 실행용 쓰레드 시작
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()

        self.get_logger().info(f"YOLODepthSubscriber started. "
                               f"RGB topic: {RGB_TOPIC}, Depth topic: {DEPTH_TOPIC}")

    # -------------------------------
    # 콜백: RGB 이미지
    # -------------------------------
    def rgb_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 큐가 비어있을 때만 새 이미지 넣기 (처리 속도 > 입력 속도 대비)
            if not self.image_queue.full():
                self.image_queue.put(img)
        except Exception as e:
            self.get_logger().error(f"RGB Image conversion failed: {e}")

    # -------------------------------
    # 콜백: Depth 이미지
    # -------------------------------
    def depth_callback(self, msg: Image):
        try:
            # depth는 보통 encoding='16UC1' (mm) 또는 '32FC1' (m)
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if depth is None:
                self.get_logger().warn('Failed to convert depth image')
                return

            if depth.ndim != 2:
                self.get_logger().warn(f"Depth image is not single-channel: shape={depth.shape}")
                return

            self.latest_depth = depth
        except Exception as e:
            self.get_logger().error(f"Depth Image conversion failed: {e}")

    # -------------------------------
    # YOLO 추론 루프 (쓰레드)
    # -------------------------------
    def detection_loop(self):
        """
        image_queue에서 RGB 이미지 하나씩 꺼내서 YOLO 추론,
        latest_depth와 함께 바운딩 박스마다 중심 픽셀 depth 계산.
        """
        while not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)
            except:
                # 일정 시간 동안 이미지가 안 들어오면 다시 루프
                continue

            # YOLO 추론
            try:
                # 필요하면 device='cuda' 추가 (GPU 사용 가능할 때만)
                # results = self.model.predict(img, stream=True, device='cuda')
                results = self.model.predict(img, stream=True)
            except Exception as e:
                self.get_logger().error(f"YOLO predict error: {e}")
                continue

            # 시각화용 이미지 복사
            vis_img = img.copy()

            # 각 detection 결과 처리
            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue

                # 각 bounding box 순회
                for box in r.boxes:
                    # 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0]) if box.cls is not None else 0
                    conf = float(box.conf[0]) if box.conf is not None else 0.0

                    # 중심 좌표 (cx, cy)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # 기본 라벨 (클래스 + confidence)
                    label = f"{self.classNames[cls]} {conf:.2f}"

                    # Depth 정보 읽기
                    depth_text = "depth: N/A"
                    distance_m = None

                    if self.latest_depth is not None:
                        depth_img = self.latest_depth
                        h, w = depth_img.shape[:2]

                        # 범위 안으로 clamp
                        u = max(0, min(w - 1, cx))
                        v = max(0, min(h - 1, cy))

                        depth_val = depth_img[v, u]

                        # float32 NaN 처리
                        if isinstance(depth_val, float) and math.isnan(depth_val):
                            distance_m = float('nan')
                        else:
                            # 16UC1 (mm) 가정: mm -> m
                            # 만약 32FC1(m) 이면 이 줄을 수정해야 함.
                            if depth_img.dtype == np.uint16:
                                distance_m = float(depth_val) / 1000.0
                            else:
                                # float32(m) 인 경우 그대로 사용
                                distance_m = float(depth_val)

                        if distance_m is not None and not math.isnan(distance_m):
                            depth_text = f"{distance_m:.3f} m"
                            # 로그 출력
                            self.get_logger().info(
                                f"Detected {self.classNames[cls]} "
                                f"center=({cx}, {cy}), depth={distance_m:.3f} m"
                            )
                        else:
                            depth_text = "depth: NaN"

                    # 화면에 박스/텍스트 그리기
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(vis_img, (cx, cy), 4, (0, 255, 255), -1)

                    # 첫 줄: 클래스 + conf
                    cv2.putText(
                        vis_img, label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    # 둘째 줄: depth 정보
                    cv2.putText(
                        vis_img, depth_text,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )

            # 결과 화면 표시
            cv2.imshow("YOLO + Depth (TurtleBot4)", vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via 'q'")
                self.should_shutdown = True
                break


def main():
    # ======= YOLO 모델 로드 =======
    # 실행 시 터미널에서 .pt 경로 입력 받기 (기존 코드 스타일 유지)
    model_path = input("Enter path to YOLO model file (.pt): ").strip()

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        sys.exit(1)

    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        model = YOLO(model_path)
    elif suffix in ['.onnx', '.engine']:
        model = YOLO(model_path, task='detect')
    else:
        print(f"Unsupported model format: {suffix}")
        sys.exit(1)

    rclpy.init()
    node = YOLODepthSubscriber(model)

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    main()
