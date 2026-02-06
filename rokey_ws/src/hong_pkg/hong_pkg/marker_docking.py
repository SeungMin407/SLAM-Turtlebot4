from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import time
import math
from nav_msgs.msg import Odometry

from .enums.robot_state import RobotState

class Nav2StatusWatcherAruco:
    def __init__(self, node, robot_id):
        # --- 1. 설정 파라미터 ---
        self.declare_parameter('image_topic', f'robot{robot_id}/oakd/rgb/image_raw/compressed') # 카메라 영상 토픽
        self.declare_parameter('cmd_vel_topic', f'robot{robot_id}/cmd_vel') # 로봇 속도 제어 토픽
        self.declare_parameter('marker_size_px', 200) # 정지할 마커의 목표 크기 (픽셀 단위, 이 값에 도달하면 정지)

        # # 회전 및 다음 목적지 설정
        self.rotation_duration = 3.0   # 180도 회전 시간 (초)
        self.forward_duration = 4.0    # 회전 후 전진할 시간 (초)
        self.forward_speed = 0.15      # 전진 속도 (m/s)
        
        # --- 2. 상태 변수 ---
        self.bridge = CvBridge() # ROS 이미지 <-> OpenCV 변환
        self.state = RobotState.ALIGNING_START # WAITING_NAV2 -> ALIGNING -> WAITING -> ROTATING -> GOING_TO_GOAL -> DONE
        self.rotation_start_time = None
        self.wait_start_time = None # 5초 대기를 위한 시간 변수
        self.forward_start_time = None  # 전진 시작 시간 저장용
        
        # ArUco 마커 설정 (6x6 딕셔너리)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        img_topic = self.get_parameter('image_topic').value
        vel_topic = self.get_parameter('cmd_vel_topic').value

        # 압축 이미지 구독
        self.image_sub = self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, vel_topic, 10)
        self.get_logger().info("Nav2 감시 및 ArUco 정렬 노드가 실행되었습니다.")

    def image_callback(self, msg):
        """카메라 영상을 처리하여 로봇을 움직이는 핵심 함수"""
        # Nav2가 도착한 상태가 아니라면 아무것도 하지 않음
        if self.state == RobotState.ALIGNING_START:
            print("aligning start")
            return
        
        corners, ids = [], None

        # ROS 이미지를 OpenCV 형식으로 변경
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape
        center_x = w / 2 # 화면의 가로 중앙 지점 (기준점)

        # 마커 탐지 (네 모서리 점(좌표), 마커 번호)
        twist = Twist()

        # [상태 1: 마커 정렬 및 도킹]
        if self.state == RobotState.ALIGNING:
            corners, ids, _ = self.detector.detectMarkers(frame)

            if ids is not None:
                # --- 여러 마커 중 화면 중앙에 가장 가까운 마커 찾기 ---
                best_idx = 0
                min_dist = float('inf')
                for i in range(len(ids)):
                    # 각 마커의 가로 중앙점 계산
                    # corners[i]는 i번째 마커의 네 모퉁이 좌표입니다.
                    # 그 네 점의 x좌표 평균을 내면 마커의 '가로 중심점'이 나옵니다.
                    m_center_x = np.mean(corners[i][0][:, 0])
                    # 화면 중앙(center_x)에서 이 마커가 얼마나 떨어져 있는지 계산
                    dist_from_center = abs(center_x - m_center_x)
                    # 화면 중앙과 가장 가까운 마커를 현재 타겟으로 설정 (# 가장 짧은 거리를 업데이트)
                    if dist_from_center < min_dist:
                        min_dist = dist_from_center
                        best_idx = i

                # 타겟 마커 정보 추출
                c = corners[best_idx][0] # 마커의 [x, y] 좌표
                marker_x = np.mean(c[:, 0]) # 마커의 가로 위치
                marker_width = np.linalg.norm(c[0] - c[1]) # 마커의 화면상 크기(너비) (첫 번째 모서리(왼쪽 위)와 두 번째 모서리(오른쪽 위)의 직선거리)

                # 현재 마커 크기를 로그로 출력 (디버깅용)
                self.get_logger().info(f"정렬 중.. 현재 마커 크기: {marker_width:.1f} / 목표: {self.get_parameter('marker_size_px').value}", once=False)

                # [1. 회전 제어] 마커를 화면 중앙으로 맞춤
                error_x = center_x - marker_x
                if abs(error_x) > 15: # 오차가 15픽셀보다 크면 회전
                    twist.angular.z = 0.002 * error_x
                else:
                    twist.angular.z = 0.0

                # [2. 전진 제어] 마커가 중앙에 어느 정도 들어오면 전진
                if abs(error_x) < 40:
                    target_size = self.get_parameter('marker_size_px').value
                    if marker_width < target_size:
                        twist.linear.x = 0.08 # 0.07m/s 속도로 전진
                    else:
                        twist.linear.x = 0.0
                        self.get_logger().info("정렬 및 정지 완료! 5초간 대기", once=True)
                        self.state = RobotState.WAITING
                        self.wait_start_time = time.time()

                self.cmd_vel_pub.publish(twist)
            else:
                # 마커가 안 보이면 제자리에 정지
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(Twist())

        # [5초 대기]
        elif self.state == RobotState.WAITING:
            if time.time() - self.wait_start_time < 5.0:
                self.cmd_vel_pub.publish(Twist()) # 멈춤 상태 유지
            else:
                self.get_logger().info("5초 대기 종료! 180도 회전을 시작합니다.")
                self.state = RobotState.ROTATING
                self.rotation_start_time = time.time()

        # [상태 2: 180도 회전]
        elif self.state == RobotState.ROTATING:
            if time.time() - self.rotation_start_time < self.rotation_duration:
                twist.angular.z = 1.05 # 회전 속도
                self.cmd_vel_pub.publish(twist)
            else:
                # 회전 완료 -> 다음 목적지 전송
                self.cmd_vel_pub.publish(Twist()) # 잠시 정지
                self.forward_start_time = time.time()
                self.get_logger().info("회전 완료! 목표 좌표로 직접 이동합니다.")
                self.state = RobotState.GOING_TO_GOAL

        # [단계 3: 목표 좌표로 직선 이동 (Nav2 미사용)]
        elif self.state == RobotState.GOING_TO_GOAL:
            # [수정] 혹시 모를 None 체크 추가
            if self.forward_start_time is None:
                self.forward_start_time = time.time()
                return
            
            elapsed_time = time.time() - self.forward_start_time
            if elapsed_time < self.forward_duration:
                twist.linear.x = self.forward_speed
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist()) # 정지
                self.state = RobotState.DONE
                self.get_logger().info("전진 완료! 모든 미션을 마칩니다.")

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Aruco Alignment", frame)
        cv2.waitKey(1)