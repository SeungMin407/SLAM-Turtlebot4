import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class Nav2StatusWatcherAruco(Node):
    def __init__(self):
        super().__init__('nav2_status_watcher_aruco')
        
        # --- 1. 설정 파라미터 ---
        # 카메라 영상 토픽
        self.declare_parameter('image_topic', 'robot4/oakd/rgb/preview/image_raw')
        # 로봇 속도 제어 토픽
        self.declare_parameter('cmd_vel_topic', 'robot4/cmd_vel')
        # 정지할 마커의 목표 크기 (픽셀 단위, 이 값에 도달하면 정지)
        self.declare_parameter('marker_size_px', 200)
        
        # --- 2. 상태 변수 ---
        self.is_nav2_finished = False   # Nav2 도착 완료 여부
        self.last_status = None         # 이전 Nav2 상태 저장
        self.bridge = CvBridge()        # ROS 이미지 <-> OpenCV 변환
        
        # ArUco 마커 설정 (6x6 딕셔너리)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # --- 3. 구독 및 발행 설정 ---
        # Nav2 액션 상태 구독 (성공 여부 확인용)
        self.status_sub = self.create_subscription(
            GoalStatusArray, 
            '/robot4/navigate_to_pose/_action/status', 
            self.nav2_status_callback, 
            10)
            
        img_topic = self.get_parameter('image_topic').value
        vel_topic = self.get_parameter('cmd_vel_topic').value

        self.image_sub = self.create_subscription(Image, img_topic, self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, vel_topic, 10)

        self.get_logger().info("Nav2 감시 및 ArUco 정렬 노드가 실행되었습니다.")

    def nav2_status_callback(self, msg):
        """Nav2가 목적지에 도착했는지 실시간으로 확인하는 함수"""
        if not msg.status_list:
            return

        current_status = msg.status_list[-1].status
        
        # Status 4 = SUCCEEDED (성공적으로 도착)
        if current_status == 4 and self.last_status != 4:
            self.get_logger().info("Nav2 목표 도착! 이제 마커 정렬을 시작합니다.")
            self.is_nav2_finished = True
        
        # 새로운 목표가 생겨서 이동 중(1, 2, 3)일 때는 정렬 모드 해제
        elif current_status in [1, 2, 3]:
            if self.is_nav2_finished:
                self.is_nav2_finished = False
        
        self.last_status = current_status

    def image_callback(self, msg):
        """카메라 영상을 처리하여 로봇을 움직이는 핵심 함수"""
        # Nav2가 도착한 상태가 아니라면 아무것도 하지 않음
        if not self.is_nav2_finished:
            return

        # ROS 이미지를 OpenCV 형식으로 변경
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape
        center_x = w / 2  # 화면의 가로 중앙 지점 (기준점)

        # 마커 탐지 (네 모서리 점(좌표), 마커 번호)
        corners, ids, _ = self.detector.detectMarkers(frame)
        twist = Twist()

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
            c = corners[best_idx][0]                   # 마커의 [x, y] 좌표
            marker_x = np.mean(c[:, 0])                # 마커의 가로 위치
            marker_width = np.linalg.norm(c[0] - c[1]) # 마커의 화면상 크기(너비) (첫 번째 모서리(왼쪽 위)와 두 번째 모서리(오른쪽 위)의 직선거리)

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
                    twist.linear.x = 0.07 # 0.07m/s 속도로 전진
                else:
                    twist.linear.x = 0.0
                    self.get_logger().info("정렬 및 정지 완료!", once=True)

            self.cmd_vel_pub.publish(twist)
            
        else:
            # 마커가 안 보이면 제자리에 정지
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

        # 화면에 마커 표시 (디버깅용)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Aruco Alignment", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = Nav2StatusWatcherAruco()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
