import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int32, Int32MultiArray, Bool # MultiArray 추가
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions

from threading import Lock, Thread
from enum import Enum, auto
import time
from collections import deque

# 모듈 import
from .modules.main_processor import MainProcessor
from .utils.nav_util import NavProcessor
from .modules.drive_commander import DriveCommander
from .enums.robot_state import RobotState

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        self.lock = Lock()

        # val
        self.battery_percent = 1.0
        self.state = RobotState.CHARGING
        
        # 라인별 상자 갯수
        self.line1_count = 0
        self.line2_count = 0
        
        # 라인별 작업 상태
        self.line_status = {1: False, 2: False}
        self.start = False
        self.start2 = False

        # [추가] 취소 조건 플래그
        self.cancel_condition = False
        
        # 로봇 별 아이디 설정
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

        # -------------------------------------------------------------------
        # [추가] QR 및 이동 관련 설정 (TestNode에서 가져옴)
        # -------------------------------------------------------------------
        self.qr_queue = deque(maxlen=10)
        
        # QR ID별 목표 좌표 맵
        self.qr_goal_map = {
            1: [(-1.58, -1.45),
                (-1.59, -0.47),
                (-1.53, 0.85),
                (-2.29, 2.47),
                (-2.92, 2.40),
                (-5.05, 2.67),
                (-5.04, 1.69),
                (-4.73, 0.75)],
            2: [(-1.58, -1.45),
                (-1.59, -0.47),
                (-1.53, 0.85),
                (-2.29, 2.47),
                (-2.92, 2.40),
                (-5.05, 2.67),
                (-5.04, 1.69)],
            3: [(-1.58, -1.45),
                (-1.59, -0.47),
                (-1.53, 0.85),
                (-2.29, 2.47),
                (-2.92, 2.40),
                (-5.05, 2.67)],}

        self.qr_goal_map2 = {
            1: [(-2.90, -1.67),
                (-2.88, -0.47),
                (-2.82, 0.13),
                (-2.93, 0.75),
                (-4.73, 0.75)],
            2: [(-2.90, -1.67),
                (-2.88, -0.47),
                (-2.82, 0.13),
                (-2.93, 0.75),
                (-4.73, 0.75),
                (-5.04, -5.05)],
            3: [(-2.90, -1.67),
                (-2.88, -0.47),
                (-2.82, 0.13),
                (-2.93, 0.75),
                (-4.73, 0.75),
                (-5.04, -5.05),
                (-5.05, 2.67)],
        }
        self.wait_point = (-2.92, 2.40)
        self.wait_point2 = (-2.93, 0.75)

        # QR 코드 구독
        self.qr_id1_sub = self.create_subscription(
            Int32, '/qr_code_id1', self.qr_callback, qos_profile_sensor_data
        )

        self.qr_id2_sub = self.create_subscription(
            Int32, '/qr_code_id2', self.qr2_callback, qos_profile_sensor_data
        )

        # 상대방 로봇 작업 상태 구독
        self.working_sub = self.create_subscription(
            Int32, self.target_working_topic, self.working_callback, qos_profile_sensor_data
        )

        # 내 작업 상태 발행
        self.work_pub = self.create_publisher(Int32, self.my_working_topic, 10)
        # -------------------------------------------------------------------

        # 배터리 상태 sub
        self.battery_state_subscriber = self.create_subscription(
            BatteryState,
            '/battery_state',
            self.battery_state_callback,
            qos_profile_sensor_data
        )
        
        # 라인1 박스 총 갯수
        self.line1_total_subscriber = self.create_subscription(
            Int32,
            '/line1/count_total',
            self.line1_total_callback,
            1
        )

        # 라인2 박스 총 갯수
        self.line2_total_subscriber = self.create_subscription(
            Int32,
            '/line2/count_total',
            self.line2_total_callback,
            1
        )

        # 라인별 작업 상태 체크
        self.line_status_subscriber = self.create_subscription(
            Int32MultiArray,
            '/line_status',
            self.line_status_callback,
            1
        )

        # 시작 알림
        self.start_subscriber = self.create_subscription(
            Bool,
            '/box_placed1',
            self.start_callback,
            1
        )

        self.start_subscriber2 = self.create_subscription(
            Bool,
            '/box_placed2',
            self.start_callback2,
            1
        )

        self.timer = self.create_timer(0.1, self.main_controller)

    # -----------------------------------------------------------
    # [추가] 콜백 함수들
    # -----------------------------------------------------------

    def qr_callback(self, msg: Int32):
        with self.lock:
            qr_id = int(msg.data)
            self.qr_queue.append(qr_id)
            self.get_logger().info(f"QR Received: {qr_id} -> Queue Size: {len(self.qr_queue)}")

    def qr2_callback(self, msg: Int32):
        with self.lock:
            qr_id = int(msg.data)
            self.qr_queue.append(qr_id)
            self.get_logger().info(f"QR Received: {qr_id} -> Queue Size: {len(self.qr_queue)}")

    def working_callback(self, msg: Int32):
        # 상대방 로봇 상태 확인
        val = int(msg.data)
        if val in (1, 2, 3): # 상대방이 작업 중이면
            self.cancel_condition = True
        elif val == 4: # 상대방 작업 완료
            self.cancel_condition = False
        # self.get_logger().info(f"Cancel Condition Updated: {self.cancel_condition}")

    def start_callback(self, msg):
        with self.lock:
            self.start = msg.data

    def start_callback2(self, msg):
        with self.lock:
            self.start2 = msg.data

    # 데이터가 비어 있지 않르면 라인 상태 저장
    def line_status_callback(self, msg):
        with self.lock:
            # [0,0,0], [0,1,0], [0,0,1], [0,1,1]
            if len(msg.data) >= 3:
                self.line_status[1] = bool(msg.data[1])
                self.line_status[2] = bool(msg.data[2])

    # 라인1의 박스 총 갯수
    def line1_total_callback(self, msg):
        with self.lock:
            self.line1_count = msg.data
            self.get_logger().info(f'line1 data: {self.line1_count}')
    
    # 라인2의 박스 총 갯수
    def line2_total_callback(self, msg):
        with self.lock:
            self.line2_count = msg.data
            self.get_logger().info(f'line2 data: {self.line2_count}')

    def battery_state_callback(self, batt_msg: BatteryState):
        with self.lock:
            self.battery_percent = batt_msg.percentage
            self.get_logger().info(f'Battery: {self.battery_percent:.2f}%')

    # 메인 루프
    def main_controller(self):
        if self.state == RobotState.CHARGING:
            if self.line1_count > 0 and self.my_robot_id == 4:
                self.nav.undock()
                while not self.nav.navigator.isTaskComplete():
                    time.sleep(0.1)

                self.state = RobotState.MOVE_TO_PICKUP
            
            if self.line2_count > 0 and self.my_robot_id == 5:
                self.nav.undock()
                while not self.nav.navigator.isTaskComplete():
                    time.sleep(0.1)
                    
                self.state = RobotState.MOVE_TO_PICKUP

        elif self.state == RobotState.MOVE_TO_PICKUP:
            if self.my_robot_id == 4:
                goal = [[-1.59,-0.47], [-1.61, -1.70]]
                self.follow_move_and_wait(goal)

            if self.my_robot_id == 5:
                goal = [[-1.53, 0.85], [-2.88, -0.47], [-2.87, -1.66]]
                self.follow_move_and_wait(goal)

            self.state = RobotState.WAITTING

        elif self.state == RobotState.WAITTING:
            with self.lock:
                current_battery = self.battery_percent
                q1 = self.line1_count
                q2 = self.line2_count
                current_status = self.line_status.copy()

            self.state = self.battery_proc.pick_up_waiting(
                1.0,
                q1,
                q2,
                current_status
            )
            self.state = RobotState.LOADING

        elif self.state == RobotState.LOADING:
            if self.start:
                self.state = RobotState.MOVE_TO_DEST
            else:
                self.state = RobotState.LOADING
                self.get_logger().info(f'작업 대기중')
            
            if self.start2:
                self.state = RobotState.MOVE_TO_DEST
            else:
                self.state = RobotState.LOADING
                self.get_logger().info(f'작업 대기중')

        elif self.state == RobotState.MOVE_TO_DEST:
            if self.my_robot_id == 4:
                self.check_detect(self.qr_goal_map)
            else:
                self.check_detect(self.qr_goal_map2)

        elif self.state == RobotState.STOP:
            pass
        elif self.state == RobotState.DONE:
            print('ssssssssssssssssssss')
        elif self.state == RobotState.DOCKING:
            self.docking_wait()
            self.state = RobotState.CHARGING

    def check_detect(self, qr_goal_map):
        # 1. 큐 확인
        if len(self.qr_queue) == 0:
            self.get_logger().warn("목적지 이동 단계지만 QR 큐가 비어있습니다. WAITTING으로 복귀합니다.")
            self.state = RobotState.WAITTING
            return

        # 2. QR 꺼내기 및 목표 설정
        qr_id = self.qr_queue.popleft()
        goal_coords = qr_goal_map.get(qr_id)

        if not goal_coords:
            self.get_logger().warn(f"알 수 없는 QR ID: {qr_id}. 스킵합니다.")
            return

        self.get_logger().info(f"QR {qr_id} 목표로 이동 시작")
        goal_array = qr_goal_map[qr_id]
        if self.my_robot_id == 4:
            Thread(target=self.move_to_goal, args=(goal_array, qr_id, self.wait_point), daemon=True).start()
        else:
            Thread(target=self.move_to_goal, args=(goal_array, qr_id, self.wait_point2), daemon=True).start()
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
        self.state = RobotState.DONE

    def follow_move_and_wait(self, goal_array):
        self.nav.way_point_no_ori(goal_array = goal_array)
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
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()