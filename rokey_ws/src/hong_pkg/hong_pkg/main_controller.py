import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int32, Int32MultiArray, Bool # MultiArray 추가
from nav2_simple_commander.robot_navigator import TaskResult

from threading import Lock
from enum import Enum, auto
import time

# 모듈 import
from .modules.main_processor import MainProcessor
from .utils.nav_util import NavProcessor
from .modules.drive_commander import DriveCommander
from .enums.robot_state import RobotState

# /box_placed
# std_msgs/msg/Bool

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        self.lock = Lock()

        # val
        self.battery_percent = 1.0
        self.state = RobotState.CHARGING
        
        # 라인별 상자 갯수
        self.line1_count = 2
        self.line2_count = 3
        
        # 라인별 작업 상태
        self.line_status = {1: False, 2: False}
        self.start = False
        
        # 로봇 별 아이디 설정
        ns = self.get_namespace()
        self.get_logger().info(ns)
        if ns == "/robot4":
            self.my_robot_id = 4
        else:
            self.my_robot_id = 5

        self.battery_proc = MainProcessor(self.my_robot_id)
        self.nav = NavProcessor()
        self.drive = DriveCommander(self)

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
            '/box_placed',
            self.start_callback,
            1
        )

        self.timer = self.create_timer(0.1, self.main_controller)
    
    def start_callback(self, msg):
        with self.lock:
            self.start = msg.data

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
            self.get_logger().info(f'data: {self.line1_count}')
    
    # 라인2의 박스 총 갯수
    def line2_total_callback(self, msg):
        with self.lock:
            self.line2_count = msg.data
            self.get_logger().info(f'data: {self.line2_count}')

    # 배터리 상태 float 값으로 받아옴
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
                current_battery,
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

        elif self.state == RobotState.MOVE_TO_DEST:
            self.get_logger().info(f'목적지 출발')
            self.state = RobotState.WAITTING
        elif self.state == RobotState.DOCKING:
            self.docking_wait()
            self.state = RobotState.CHARGING

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