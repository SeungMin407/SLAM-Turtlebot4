import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int32, Int32MultiArray, Bool
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions

# [필수] 콜백 그룹 및 스레드 관련 임포트
from rclpy.callback_groups import ReentrantCallbackGroup
from threading import Lock, Thread # Thread 필수
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

import sys
import termios
import tty

from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration

AUDIO_TOPIC = '/robot4/cmd_audio'
CMD_VEL_TOPIC = '/robot4/cmd_vel'
from geometry_msgs.msg import Twist

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        self.lock = Lock()
        
        # 데드락 방지용 콜백 그룹 (필수)
        self.callback_group = ReentrantCallbackGroup()

        self.em_stop_sub = self.create_subscription(
        Bool,
        '/emergency_stop',
        self.emergency_stop_cb,
        10,
        callback_group=self.callback_group)

        # ---------------------------------------------------
        # [1] Main Controller 변수 초기화
        # ---------------------------------------------------
        self.battery_percent = 1.0
        self.state = RobotState.CHARGING  # 메인 상태
        self.is_helping = False  # 지원 모드 플래그
        self.is_shutdown = False # 종료 플래그
        
        self.line1_count = 0
        self.line2_count = 0
        self.line_status = {1: False, 2: False}
        self.start = False
        self.start2 = False
        self.cancel_condition4 = False
        self.cancel_condition5 = False
        
        ns = self.get_namespace()
        self.get_logger().info(ns)

        if ns == "/robot4":
            self.my_robot_id = 4
            self.current_working_line = 1
            self.my_working_topic = '/robot4/working'
            self.target_working_topic = '/robot5/working'
        else:
            self.my_robot_id = 5
            self.current_working_line = 2
            self.my_working_topic = '/robot5/working'
            self.target_working_topic = '/robot4/working'

        self.battery_proc = MainProcessor(self.my_robot_id)
        self.nav = NavProcessor()
        self.drive = DriveCommander(self)

        # 큐를 2개로 분리 (내 것, 남의 것)
        self.queue1 = deque(maxlen=10) # 1번 라인용
        self.queue2 = deque(maxlen=10) # 2번 라인용
        
        # QR Goal Maps (생략 없이 그대로 유지)
        self.qr_goal_map = [(-1.61, -1.70), (-2.29, 2.47), (-2.92, 2.40)]
            
        # [맵 이름 변경 없음 - 기존 코드 호환성 유지]
        self.goal_map = { # 1번 라인 맵
            1: [(-4.73, 0.75)],
            2: [(-5.04, 1.69)],
            3: [(-5.05, 2.67)]
        }
        self.qr_goal_map2 = [(-2.87, -1.66), (-2.93, 0.75)]

        self.goal_map2 = { # 2번 라인 맵
            1: [(-4.73, 0.75)],
            2: [(-5.04, 1.69)],
            3: [(-5.05, 2.67)]
        }

        self.final_map = [[-2.92, 2.40], [-2.29, 2.47], [-1.61, -1.70]]
        self.final_map2 = [[-2.93, 0.75], [-2.87, -1.66]]
        self.wait_point = (-2.92, 2.40)
        self.wait_point2 = (-2.93, 0.75)

        # ---------------------------------------------------
        # [2] ArUco 관련 변수 초기화
        # ---------------------------------------------------
        self.declare_parameter('image_topic', f'{ns}/oakd/rgb/image_raw/compressed')
        self.declare_parameter('cmd_vel_topic', f'{ns}/cmd_vel')
        self.declare_parameter('marker_size_px', 200)

        self.rotation_duration = 3.0
        self.forward_duration = 4.0
        self.forward_speed = 0.15
        
        self.image_start = False 
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

        # 모든 Subscriber에 callback_group 적용 (이동 중 수신 위해 필수)
        self.image_sub = self.create_subscription(CompressedImage, img_topic, self.image_callback, 10, callback_group=self.callback_group)
        self.cmd_vel_pub = self.create_publisher(Twist, vel_topic, 10)
        
        # QR 코드 구독 분리 (1번, 2번)
        self.qr_id1_sub = self.create_subscription(Int32, '/qr_code_id1', self.callback_qr1, qos_profile_sensor_data, callback_group=self.callback_group)
        self.qr_id2_sub = self.create_subscription(Int32, '/qr_code_id2', self.callback_qr2, qos_profile_sensor_data, callback_group=self.callback_group)
        
        self.pub_pop_queue1 = self.create_publisher(Int32, '/pop_queue1', 10)
        self.pub_pop_queue2 = self.create_publisher(Int32, '/pop_queue2', 10)
        self.sub_pop_queue1 = self.create_subscription(Int32, '/pop_queue1', self.callback_pop_queue1, 10, callback_group=self.callback_group)
        self.sub_pop_queue2 = self.create_subscription(Int32, '/pop_queue2', self.callback_pop_queue2, 10, callback_group=self.callback_group)

        self.working_sub4 = self.create_subscription(Int32, '/robot5/working', self.working_callback4, 10, callback_group=self.callback_group)
        self.working_sub5 = self.create_subscription(Int32, '/robot4/working', self.working_callback5, 10, callback_group=self.callback_group)

        self.battery_state_subscriber = self.create_subscription(BatteryState, '/battery_state', self.battery_state_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.line1_total_subscriber = self.create_subscription(Int32, '/line1/count_total', self.line1_total_callback, 1, callback_group=self.callback_group)
        self.line2_total_subscriber = self.create_subscription(Int32, '/line2/count_total', self.line2_total_callback, 1, callback_group=self.callback_group)
        self.line_status_subscriber = self.create_subscription(Int32MultiArray, '/line_status', self.line_status_callback, 1, callback_group=self.callback_group)
        self.start_subscriber = self.create_subscription(Bool, '/box_placed1', self.start_callback, 1, callback_group=self.callback_group)
        self.start_subscriber2 = self.create_subscription(Bool, '/box_placed2', self.start_callback2, 1, callback_group=self.callback_group)

        self.work_pub4 = self.create_publisher(Int32, '/robot4/working', 10)
        self.work_pub5 = self.create_publisher(Int32, '/robot5/working', 10)


        self.get_logger().info("MainController Ready.")
        
        # 타이머 삭제 -> 스레드 시작 (ValueError 해결용)
        # self.timer = self.create_timer(0.1, self.main_controller) <--- 삭제
        self.logic_thread = Thread(target=self.run_logic_loop, daemon=True)
        self.logic_thread.start()
    def emergency_stop_cb(self, msg: Bool):     
        if not msg.data:
            return

        self.get_logger().fatal("🚨 EMERGENCY STOP received -> cancel nav + stop")

        with self.lock:
            self.is_shutdown = True
            self.state = RobotState.STOP

        # 1) Nav2 waypoint/action cancel
        try:
            self.nav.navigator.cancelTask()
        except Exception as e:
            self.get_logger().warn(f"cancelTask failed: {e}")

        # 2) 즉시 정지 cmd_vel (여러 번 쏘면 더 확실)
        stop = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(stop)
            time.sleep(0.02)
    # 로직 루프 (타이머 대체)
    def run_logic_loop(self):
        time.sleep(2.0) # 안정화 대기
        while rclpy.ok():
            try:
                self.main_controller()
            except Exception as e:
                self.get_logger().error(f"Error: {e}")
            time.sleep(0.1)

    # 상대 큐 삭제
    def callback_pop_queue1(self, msg: Int32):
        with self.lock:
            if len(self.queue1) > 0:
                removed = self.queue1.popleft() # 같이 삭제
                self.get_logger().info(f"🔄 동기화: 상대방이 처리하여 Queue1에서 {removed} 삭제됨")

    def callback_pop_queue2(self, msg: Int32):
        with self.lock:
            if len(self.queue2) > 0:
                removed = self.queue2.popleft() # 같이 삭제
                self.get_logger().info(f"🔄 동기화: 상대방이 처리하여 Queue2에서 {removed} 삭제됨")

    # QR 콜백 분리
    def callback_qr1(self, msg: Int32):
        with self.lock:
            self.queue1.append(int(msg.data)) # 1번 큐에 저장
            self.get_logger().info(f"QR1: {msg.data}")

    def callback_qr2(self, msg: Int32):
        with self.lock:
            self.queue2.append(int(msg.data)) # 2번 큐에 저장
            self.get_logger().info(f"QR2: {msg.data}")

    # -----------------------------------------------------------
    # [콜백 함수들] (기존 코드 그대로 유지)
    # -----------------------------------------------------------
    def image_callback(self, msg):
        if not self.image_start: return
        if self.aruco_state == RobotState.DONE: return
        corners, ids = [], None
        try: frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception: return
        h, w, _ = frame.shape
        center_x = w / 2
        twist = Twist()

        if self.aruco_state == RobotState.ALIGNING:
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is not None:
                c = corners[0][0]
                marker_x = np.mean(c[:, 0])
                marker_width = np.linalg.norm(c[0] - c[1])
                error_x = center_x - marker_x
                if abs(error_x) > 15: twist.angular.z = 0.002 * error_x
                else: twist.angular.z = 0.0
                if abs(error_x) < 40:
                    target_size = self.get_parameter('marker_size_px').value
                    if marker_width < target_size: twist.linear.x = 0.08
                    else:
                        twist.linear.x = 0.0
                        with self.lock:
                            self.aruco_state = RobotState.FIVE_WAITTING
                            self.wait_start_time = time.time()
                self.cmd_vel_pub.publish(twist)
            else:
                twist.linear.x, twist.angular.z = 0.0, 0.0
                self.cmd_vel_pub.publish(twist)

        elif self.aruco_state == RobotState.FIVE_WAITTING:
            if time.time() - self.wait_start_time < 5.0: self.cmd_vel_pub.publish(Twist())
            else:
                with self.lock:
                    self.aruco_state = RobotState.ROTATING
                    self.rotation_start_time = time.time()

        elif self.aruco_state == RobotState.ROTATING:
            if time.time() - self.rotation_start_time < self.rotation_duration:
                twist.angular.z = 1.05
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                with self.lock:
                    self.forward_start_time = time.time()
                    self.aruco_state = RobotState.GOING_TO_GOAL

        elif self.aruco_state == RobotState.GOING_TO_GOAL:
            if self.forward_start_time is None: self.forward_start_time = time.time()
            if time.time() - self.forward_start_time < self.forward_duration:
                twist.linear.x = self.forward_speed
                self.cmd_vel_pub.publish(twist)
            else:
                self.cmd_vel_pub.publish(Twist())
                with self.lock: self.aruco_state = RobotState.DONE

    # (이전 qr_callback 삭제 -> callback_qr1, 2로 대체됨)

    def working_callback4(self, msg: Int32):
        with self.lock:
            val = int(msg.data)
            self.cancel_condition4 = (val in (1, 2, 3))
    
    def working_callback5(self, msg: Int32):
        with self.lock:
            val = int(msg.data)
            self.cancel_condition5 = (val in (1, 2, 3))

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
        if self.state == RobotState.CHARGING:
            if (len(self.queue1) > 0 and self.my_robot_id == 4) or \
               (len(self.queue2) > 0 and self.my_robot_id == 5):
                if self.nav.navigator.getDockedStatus():
                    print('언도킹을 시작합니다...')
                    self.nav.navigator.undock()
                    while not self.nav.navigator.isTaskComplete():
                        time.sleep(0.1)
                else:
                    print('이미 언도킹되어 있습니다.')

                self.state = RobotState.MOVE_TO_PICKUP

        elif self.state == RobotState.MOVE_TO_PICKUP:
            if self.my_robot_id == 4: self.follow_move_and_wait([[-1.59,-0.47], [-1.61, -1.70]])
            if self.my_robot_id == 5: self.follow_move_and_wait([[-1.53, 0.85], [-2.88, -0.47], [-2.87, -1.66]])
            
            
            # 복귀 시 원래 담당 라인으로 리셋
            if self.my_robot_id == 4:
                self.current_working_line = 1
            else:
                self.current_working_line = 2

            self.is_helping = False
            self.state = RobotState.WAITTING
        elif self.state == RobotState.WAITTING:
            print(f'라인 1 : {len(self.queue1)} queue size 1: {self.queue1}')
            print(f'라인 2 : {len(self.queue2)} queue size 1: {self.queue2}')
            if self.current_working_line == 1:
                self.line_status[1] = True  # 나 1번 라인 일한다!
            else:
                self.line_status[2] = True  # 나 2번 라인 일한다!
            self.drive.publish_line_status(self.line_status[1], self.line_status[2])

            with self.lock:
                q1 = len(self.queue1)
                q2 = len(self.queue2)
                curr = self.line_status.copy()
                
            if self.my_robot_id == 4:
                my_start = self.start if self.my_robot_id == 4 else self.start2
                next_state = self.battery_proc.pick_up_waiting(self.battery_percent, q1, q2, curr, my_start)
            else:
                target_start = self.start2 if self.my_robot_id == 4 else self.start
                next_state = self.battery_proc.pick_up_waiting(self.battery_percent, q2, q1, curr, target_start)
            
            self.state = next_state
            
            # 지원 모드 판단 및 라인 변경
            if self.state == RobotState.GO_TO_OTHER:
                self.is_helping = True
                if self.my_robot_id == 4:
                    self.current_working_line = 2
                    self.line_status[1] = False # 1번 라인 비움
                    self.line_status[2] = True  # 2번 라인 점유하러 감
                else:
                    self.current_working_line = 1
                    self.line_status[2] = False # 2번 라인 비움
                    self.line_status[1] = True  # 1번 라인 점유하러 감
                self.drive.publish_line_status(self.line_status[1], self.line_status[2])
            else:
                self.is_helping = False
                if self.my_robot_id == 4:
                    self.current_working_line = 1
                else:
                    self.current_working_line = 2

        elif self.state == RobotState.GO_TO_OTHER:
            target_start = self.start2 if self.my_robot_id == 4 else self.start
            if target_start:
                self.state = RobotState.MOVE_TO_DEST
                self.start = False; self.start2 = False
                if self.current_working_line == 1:
                    self.line_status[1] = False  # 나 1번 라인 일한다!
                else:
                    self.line_status[2] = False  # 나 2번 라인 일한다!
                self.drive.publish_line_status(self.line_status[1], self.line_status[2])
            else:
                self.state = RobotState.GO_TO_OTHER

        elif self.state == RobotState.LOADING:
            my_start = self.start if self.my_robot_id == 4 else self.start2
            if my_start:
                self.state = RobotState.MOVE_TO_DEST
                self.start = False; self.start2 = False
                if self.current_working_line == 1:
                    self.line_status[1] = False  # 나 1번 라인 일한다!
                else:
                    self.line_status[2] = False  # 나 2번 라인 일한다!
                self.drive.publish_line_status(self.line_status[1], self.line_status[2])
            else:
                self.state = RobotState.LOADING

        elif self.state == RobotState.MOVE_TO_DEST:
            if self.my_robot_id == 4:
                self.follow_move_and_wait(self.qr_goal_map)
            else:
                self.follow_move_and_wait(self.qr_goal_map2)
            self.state = RobotState.GO_TO_WAIT

        # [수정] 큐 선택 로직 적용
        elif self.state == RobotState.GO_TO_WAIT:
            
            # 1. 현재 작업 라인에 맞는 큐 선택
            if self.current_working_line == 1:
                target_queue = self.queue1
                target_map = self.goal_map # (기존 이름 유지: goal_map = 1번 라인용)
            else:
                target_queue = self.queue2
                target_map = self.goal_map2 # (기존 이름 유지: goal_map2 = 2번 라인용)

            # 2. 큐 처리
            if len(target_queue) > 0:
                qr_id = target_queue.popleft()

                notify_msg = Int32()
                notify_msg.data = int(qr_id) # 값은 중요치 않음, 신호가 중요
                
                if self.current_working_line == 1:
                    self.pub_pop_queue1.publish(notify_msg) # "나 1번큐 뺐어!"
                else:
                    self.pub_pop_queue2.publish(notify_msg) # "나 2번큐 뺐어!"
                
                if self.my_robot_id == 4:
                    for i in range(10):
                        self.work_pub4.publish(Int32(data=int(qr_id)))
                        time.sleep(0.1)
                    
                    # [추가] 이동 전 정지 신호 대기
                    while self.cancel_condition4:
                        time.sleep(0.1)
                    
                    if qr_id in target_map:
                        self.follow_move_and_wait(target_map[qr_id],TurtleBot4Directions.SOUTH)
                        self.state = RobotState.MOVE_ALIGNING
                    else:
                        pass # Error handling

                else:
                    for i in range(10):
                        self.work_pub5.publish(Int32(data=int(qr_id)))
                        time.sleep(0.1)
                    
                    while self.cancel_condition5:
                        time.sleep(0.1)
                    
                    if qr_id in target_map:
                        self.follow_move_and_wait(target_map[qr_id],TurtleBot4Directions.SOUTH)
                        self.state = RobotState.MOVE_ALIGNING
                    else:
                        pass
            else:
                pass

        elif self.state == RobotState.MOVE_ALIGNING:
            if not self.image_start:
                with self.lock:
                    self.image_start = True
                    self.aruco_state = RobotState.ALIGNING
            elif self.aruco_state == RobotState.DONE:
                with self.lock:
                    self.image_start = False
                    self.aruco_state = RobotState.ALIGNING_START
                self.state = RobotState.RETURN_TO_LINE

        elif self.state == RobotState.RETURN_TO_LINE:
            self.is_helping = False
            if self.my_robot_id == 4 :
                for i in range(10):
                    self.drive.robot4_send_work_finish()
                    time.sleep(0.1)
                self.follow_move_and_wait(self.final_map)
            else:
                for i in range(10):
                    self.drive.robot5_send_work_finish()
                    time.sleep(0.1)
                self.follow_move_and_wait(self.final_map2)
            self.state = RobotState.WAITTING

        elif self.state == RobotState.DOCKING:
            if not self.nav.navigator.getDockedStatus():
                print('도킹을 시작합니다...')
                self.nav.navigator.dock()
                while not self.nav.navigator.isTaskComplete():
                    time.sleep(0.1)
            else:
                print('이미 도킹되어 있습니다.')
            self.state = RobotState.CHARGING
            
        elif self.state == RobotState.STOP:
            pass

    # 이동 중 정지 기능 추가
    def follow_move_and_wait(self, goal_array, goal_or=None):
        self.nav.way_point_no_ori(goal_array=goal_array, goal_or=goal_or)
        while not self.nav.navigator.isTaskComplete():
            if self.is_shutdown: 
                self.nav.navigator.cancelTask()
                break
            time.sleep(0.1)

    def docking_wait(self):
        self.nav.dock()
        while not self.nav.navigator.isTaskComplete():
            time.sleep(0.1)

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')

        self.audio_pub = self.create_publisher(AudioNoteVector, AUDIO_TOPIC, 10)
        self.em_pub = self.create_publisher(Bool, '/emergency_stop', 10)

    def make_note(self, freq_hz: float, duration_sec: float) -> AudioNote:
        note = AudioNote()
        note.frequency = int(freq_hz)
        note.max_runtime = Duration(sec=0, nanosec=int(duration_sec * 1e9))
        return note

    def beep(self):
        msg = AudioNoteVector()
        msg.append = False
        msg.notes = [
            self.make_note(880.0, 0.2),
            self.make_note(440.0, 0.2),
        ]
        self.audio_pub.publish(msg)

    def emergency_stop(self):
        self.em_pub.publish(Bool(data=True))

def main(args=None):
    rclpy.init(args=args)

    main_node = MainController()   # ✅ 네비를 실제로 돌리는 노드
    beep_node = BeepNode()         # ✅ 키 입력 + 삐 + 비상정지 토픽

    executor = MultiThreadedExecutor()
    executor.add_node(main_node)
    executor.add_node(beep_node)

    # executor는 백그라운드 스레드로 spin
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            key = get_key()

            if key == '\x03':  # Ctrl+C (raw 모드)
                beep_node.beep()
                beep_node.emergency_stop()
                continue

            if key == 'q':
                break

            time.sleep(0.01)

    finally:
        executor.shutdown()
        main_node.destroy_node()
        beep_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()