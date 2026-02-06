import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int32, Int32MultiArray, Bool
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions

from threading import Lock, Thread
from enum import Enum, auto
import time
from collections import deque
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CompressedImage
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatusArray
from nav_msgs.msg import Odometry

# 모듈 import
from .modules.main_processor import MainProcessor
from .utils.nav_util import NavProcessor
from .modules.drive_commander import DriveCommander
from .enums.robot_state import RobotState

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        self.lock = Lock()

        # ---------------------------------------------------
        # [1] Main Controller 변수 초기화
        # ---------------------------------------------------
        self.battery_percent = 1.0
        self.state = RobotState.CHARGING  # 메인 상태
        
        self.line1_count = 0
        self.line2_count = 0
        self.line_status = {1: False, 2: False}
        self.start = False
        self.start2 = False
        self.cancel_condition = False
        self.cancel_condition2 = False
        
        ns = self.get_namespace()
        self.get_logger().info(ns)

        if ns == "/robot4":
            self.my_robot_id = 4
            self.my_working_topic = '/robot4/working'
            self.target_working_topic = '/robot5/working'
        else:
            self.my_robot_id = 5
            self.my_working_topic = '/robot5/working'
            self.target_working_topic = '/robot4/working'

        self.battery_proc = MainProcessor(self.my_robot_id)
        self.nav = NavProcessor()
        self.drive = DriveCommander(self)

        self.qr_queue = deque(maxlen=10)
        
        # QR Goal Maps (생략 없이 그대로 유지)
        self.qr_goal_map = {
            1: [(-1.58, -1.45), (-1.59, -0.47), (-1.53, 0.85), (-2.29, 2.47), (-2.92, 2.40), (-5.05, 2.67), (-5.04, 1.69), (-4.73, 0.75)],
            2: [(-1.58, -1.45), (-1.59, -0.47), (-1.53, 0.85), (-2.29, 2.47), (-2.92, 2.40), (-5.05, 2.67), (-5.04, 1.69)],
            3: [(-1.58, -1.45), (-1.59, -0.47), (-1.53, 0.85), (-2.29, 2.47), (-2.92, 2.40), (-5.05, 2.67)],
        }
        self.qr_goal_map2 = {
            1: [(-2.90, -1.67), (-2.88, -0.47), (-2.82, 0.13), (-2.93, 0.75), (-4.73, 0.75)],
            2: [(-2.90, -1.67), (-2.88, -0.47), (-2.82, 0.13), (-2.93, 0.75), (-4.73, 0.75), (-5.04, 1.69)],
            3: [(-2.90, -1.67), (-2.88, -0.47), (-2.82, 0.13), (-2.93, 0.75), (-4.73, 0.75), (-5.04, 1.69), (-5.05, 2.67)],
        }

        self.final_map = [[-2.92, 2.40],[-2.29, 2.47],[-1.53, 0.85],[-1.59, -0.47],[-1.61, -1.70]]
        self.final_map2 = [[-2.93, 0.75],[-2.88, -0.47],[-2.87, -1.66]]
        self.wait_point = (-2.92, 2.40)
        self.wait_point2 = (-2.93, 0.75)

        # ---------------------------------------------------
        # [2] ArUco 관련 변수 초기화 (Nav2StatusWatcherAruco 병합)
        # ---------------------------------------------------
        # 파라미터 선언
        self.declare_parameter('image_topic', f'{ns}/oakd/rgb/image_raw/compressed')
        self.declare_parameter('cmd_vel_topic', f'{ns}/cmd_vel')
        self.declare_parameter('marker_size_px', 200)

        # ArUco 제어 변수
        self.rotation_duration = 3.0
        self.forward_duration = 4.0
        self.forward_speed = 0.15
        
        # [수정] image_start는 기본 False (True일 때 작동하게 변경)
        self.image_start = False 
        
        # [중요] 메인 상태(self.state)와 겹치지 않게 별도 변수 사용
        self.aruco_state = RobotState.ALIGNING_START 
        
        self.bridge = CvBridge()
        self.rotation_start_time = None
        self.wait_start_time = None
        self.forward_start_time = None
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # ---------------------------------------------------
        # [3] Subscribers & Publishers
        # ---------------------------------------------------
        img_topic = self.get_parameter('image_topic').value
        vel_topic = self.get_parameter('cmd_vel_topic').value

        self.image_sub = self.create_subscription(CompressedImage, img_topic, self.image_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, vel_topic, 10)
        
        self.qr_id1_sub = self.create_subscription(Int32, '/qr_code_id1', self.qr_callback, qos_profile_sensor_data)
        self.qr_id2_sub = self.create_subscription(Int32, '/qr_code_id2', self.qr2_callback, qos_profile_sensor_data)
        self.working_sub = self.create_subscription(Int32, self.target_working_topic, self.working_callback, qos_profile_sensor_data)
        self.working_sub2 = self.create_subscription(Int32, self.target_working_topic, self.working2_callback, qos_profile_sensor_data)
        
        self.battery_state_subscriber = self.create_subscription(BatteryState, '/battery_state', self.battery_state_callback, qos_profile_sensor_data)
        self.line1_total_subscriber = self.create_subscription(Int32, '/line1/count_total', self.line1_total_callback, 1)
        self.line2_total_subscriber = self.create_subscription(Int32, '/line2/count_total', self.line2_total_callback, 1)
        self.line_status_subscriber = self.create_subscription(Int32MultiArray, '/line_status', self.line_status_callback, 1)
        self.start_subscriber = self.create_subscription(Bool, '/box_placed1', self.start_callback, 1)
        self.start_subscriber2 = self.create_subscription(Bool, '/box_placed2', self.start_callback2, 1)

        self.work_pub = self.create_publisher(Int32, self.my_working_topic, 10)

        self.get_logger().info("MainController & ArUco Logic Ready.")
        self.timer = self.create_timer(0.1, self.main_controller)

    # -----------------------------------------------------------
    # [콜백 함수들]
    # -----------------------------------------------------------

    def image_callback(self, msg):
        # [수정] image_start가 False이면 실행하지 않음 (True여야 작동)
        if not self.image_start:
            return
        
        # [수정] 작업이 완료되면 더 이상 돌지 않음
        if self.aruco_state == RobotState.DONE:
            return

        corners, ids = [], None
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        h, w, _ = frame.shape
        center_x = w / 2
        twist = Twist()

        # [상태 1: 마커 정렬]
        # [수정] self.state 대신 self.aruco_state 사용
        if self.aruco_state == RobotState.ALIGNING:
            corners, ids, _ = self.detector.detectMarkers(frame)

            if ids is not None:
                best_idx = 0
                min_dist = float('inf')
                for i in range(len(ids)):
                    m_center_x = np.mean(corners[i][0][:, 0])
                    dist_from_center = abs(center_x - m_center_x)
                    if dist_from_center < min_dist:
                        min_dist = dist_from_center
                        best_idx = i

                c = corners[best_idx][0]
                marker_x = np.mean(c[:, 0])
                marker_width = np.linalg.norm(c[0] - c[1])

                self.get_logger().info(f"정렬 중.. 크기: {marker_width:.1f}", once=True)

                error_x = center_x - marker_x
                if abs(error_x) > 15:
                    twist.angular.z = 0.002 * error_x
                else:
                    twist.angular.z = 0.0

                if abs(error_x) < 40:
                    target_size = self.get_parameter('marker_size_px').value
                    if marker_width < target_size:
                        twist.linear.x = 0.08
                    else:
                        twist.linear.x = 0.0
                        self.get_logger().info("정렬 완료! 5초 대기")
                        # [중요] 상태 변경 시 lock 사용 권장
                        with self.lock:
                            self.aruco_state = RobotState.FIVE_WAITTING
                            self.wait_start_time = time.time()

                self.cmd_vel_pub.publish(twist)
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)

        # [5초 대기]
        elif self.aruco_state == RobotState.FIVE_WAITTING:
            if time.time() - self.wait_start_time < 5.0:
                self.cmd_vel_pub.publish(Twist())
            else:
                self.get_logger().info("180도 회전 시작")
                with self.lock:
                    self.aruco_state = RobotState.ROTATING
                    self.rotation_start_time = time.time()

        # [180도 회전]
        elif self.aruco_state == RobotState.ROTATING:
            if time.time() - self.rotation_start_time < self.rotation_duration:
                twist.angular.z = 1.05
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("직진 시작")
                with self.lock:
                    self.forward_start_time = time.time()
                    self.aruco_state = RobotState.GOING_TO_GOAL

        # [마지막 직진]
        elif self.aruco_state == RobotState.GOING_TO_GOAL:
            if self.forward_start_time is None:
                self.forward_start_time = time.time()
            
            if time.time() - self.forward_start_time < self.forward_duration:
                twist.linear.x = self.forward_speed
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                self.get_logger().info("아루코 정렬 최종 완료")
                with self.lock:
                    self.aruco_state = RobotState.DONE

    def qr_callback(self, msg: Int32):
        with self.lock:
            self.qr_queue.append(int(msg.data))
            self.get_logger().info(f"QR Received: {msg.data}")

    def qr2_callback(self, msg: Int32):
        with self.lock:
            self.qr_queue.append(int(msg.data))
            self.get_logger().info(f"QR Received: {msg.data}")

    def working_callback(self, msg: Int32):
        val = int(msg.data)
        self.cancel_condition = (val in (1, 2, 3))
    
    def working2_callback(self, msg: Int32):
        val = int(msg.data)
        self.cancel_condition2 = (val in (1, 2, 3))

    def start_callback(self, msg):
        with self.lock: self.start = msg.data

    def start_callback2(self, msg):
        with self.lock: self.start2 = msg.data

    def line_status_callback(self, msg):
        with self.lock:
            if len(msg.data) >= 3:
                self.line_status[1] = bool(msg.data[1])
                self.line_status[2] = bool(msg.data[2])

    def line1_total_callback(self, msg):
        with self.lock: self.line1_count = msg.data

    def line2_total_callback(self, msg):
        with self.lock: self.line2_count = msg.data

    def battery_state_callback(self, msg: BatteryState):
        with self.lock: self.battery_percent = msg.percentage

    # -----------------------------------------------------------
    # [메인 루프]
    # -----------------------------------------------------------
    def main_controller(self):
        # 1. 충전 상태
        if self.state == RobotState.CHARGING:
            if (self.line1_count > 0 and self.my_robot_id == 4) or \
               (self.line2_count > 0 and self.my_robot_id == 5):
                self.nav.undock()
                while not self.nav.navigator.isTaskComplete():
                    time.sleep(0.1)
                self.state = RobotState.MOVE_TO_PICKUP

        # 2. 픽업 위치로 이동
        elif self.state == RobotState.MOVE_TO_PICKUP:
            if self.my_robot_id == 4:
                goal = [[-1.59,-0.47], [-1.61, -1.70]]
                self.follow_move_and_wait(goal)
            if self.my_robot_id == 5:
                goal = [[-1.53, 0.85], [-2.88, -0.47], [-2.87, -1.66]]
                self.follow_move_and_wait(goal)
            self.state = RobotState.WAITTING

        # 3. 대기
        elif self.state == RobotState.WAITTING:
            print(f'line 1 : {self.line1_count}')
            print(f'line 2 : {self.line2_count}')
            with self.lock:
                q1 = self.line1_count
                q2 = self.line2_count
                current_status = self.line_status.copy()
            
            # (참고) pick_up_waiting 내부 로직에 따라 상태 반환
            if self.my_robot_id == 4:
                # 4번 로봇: 내 라인(q1) 우선, 없으면 q2 지원
                self.state = self.battery_proc.pick_up_waiting(self.battery_percent, q1, q2, current_status)
            else:
                # 5번 로봇: 내 라인(q2) 우선, 없으면 q1 지원
                self.state = self.battery_proc.pick_up_waiting(self.battery_percent, q2, q1, current_status)

        # 4. 로딩
        elif self.state == RobotState.LOADING:
            if (self.my_robot_id == 4 and self.start) or \
               (self.my_robot_id == 5 and self.start2):
                self.state = RobotState.MOVE_TO_DEST
                self.start = False; self.start2 = False
            else:
                self.state = RobotState.LOADING

        # 5. 목적지 이동
        elif self.state == RobotState.MOVE_TO_DEST:
            if self.my_robot_id == 4:
                self.check_detect(self.qr_goal_map)
            else:
                self.check_detect(self.qr_goal_map2)

        # [중요] 7. 마커 정렬 단계
        elif self.state == RobotState.MOVE_ALIGNING:
            self.get_logger().info("QR 이동 완료. 마커 정렬 모드로 전환합니다.")
            # (A) 아루코 로직 시작 전: 플래그 켜기
            if not self.image_start:
                with self.lock:
                    self.image_start = True # 이제 image_callback이 동작함
                    self.aruco_state = RobotState.ALIGNING # 정렬 상태로 설정
                self.get_logger().info(">>> 마커 정렬 시작 (ALIGNING)")

            elif self.aruco_state == RobotState.DONE:
                self.get_logger().info(">>> 마커 정렬 완료 확인! DOCKING 이동")
                
                # 정리 (선택)
                with self.lock:
                    self.image_start = False # 다시 끄기
                    self.aruco_state = RobotState.ALIGNING_START # 리셋
                
                self.state = RobotState.RETURN_TO_LINE # 다음 상태로
            else:
                pass

        elif self.state == RobotState.RETURN_TO_LINE:
            self.get_logger().info("이동 완료 (DOCKING STATE)")
            if self.my_robot_id == 4 :
                self.get_logger().info("4번 움직여")
                self.drive.robot4_send_work_finish()
                self.follow_move_and_wait(self.final_map)
            else:
                self.get_logger().info("5번 움직여")
                self.drive.robot5_send_work_finish()
                self.follow_move_and_wait(self.final_map2)
            self.state = RobotState.WAITTING
        elif self.state == RobotState.DOCKING:
            self.docking_wait()
            self.state = RobotState.STOP
        elif self.state == RobotState.STOP:
            pass

    # -----------------------------------------------------------
    # [Helper 함수]
    # -----------------------------------------------------------
    def check_detect(self, qr_goal_map):
        if len(self.qr_queue) == 0:
            self.state = RobotState.WAITTING
            return

        qr_id = self.qr_queue.popleft()
        goal_coords = qr_goal_map.get(qr_id)

        if not goal_coords:
            return

        self.get_logger().info(f"QR {qr_id} 이동 시작")
        goal_array = qr_goal_map[qr_id]
        
        args = (goal_array, qr_id, self.wait_point if self.my_robot_id == 4 else self.wait_point2)
        Thread(target=self.move_to_goal, args=args, daemon=True).start()
        
        self.state = RobotState.STOP

    def move_to_goal(self, goal_array, qr_id, wait_point):
        def pub_work():
            msg = Int32()
            msg.data = int(qr_id)
            self.work_pub.publish(msg)
            self.get_logger().info(f"[seq1 reached] published qr_id={qr_id} to {self.my_working_topic}")

        self.nav.way_point_no_ori2(
            goal_array=goal_array,
            goal_or=TurtleBot4Directions.SOUTH,
            wait_point=wait_point,
            cancel=lambda: self.cancel_condition,
            on_reach=pub_work,
        )

        self.state = RobotState.MOVE_ALIGNING 

    def follow_move_and_wait(self, goal_array):
        self.nav.way_point_no_ori(goal_array=goal_array)
        while not self.nav.navigator.isTaskComplete():
            time.sleep(0.1)

    def docking_wait(self):
        self.nav.dock()
        while not self.nav.navigator.isTaskComplete():
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()