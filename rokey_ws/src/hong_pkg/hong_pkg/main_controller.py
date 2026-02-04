import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Int32, Int32MultiArray # MultiArray 추가

from threading import Lock
from enum import Enum, auto

# 모듈 import
from .modules.battery_processor import BatteryProcessor

class RobotState(Enum):
    CHARGING = auto()        # 도킹 스테이션에서 대기/충전
    MOVE_TO_PICKUP = auto()  # 라인으로 이동
    WAITTING = auto()        # 배터리 상태와 라인 박스 상태 체크
    LOADING = auto()         # 5초 기다리기 (박스 받는 중)
    MOVE_TO_DEST = auto()    # 목적지로 배달
    RETURN_TO_LINE = auto()  # 라인으로 복귀 (다음 작업 대기)

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')

        self.lock = Lock()

        # val
        self.battery_percent = 0.0
        self.state = RobotState.CHARGING
        
        # 라인별 상자 갯수
        self.line1_count = 3
        self.line2_count = 2
        
        # 라인별 작업 상태
        self.line_status = {1: False, 2: False}
        
        # 로봇 별 아이디 설정
        ns = self.get_namespace()
        self.my_robot_id = ns
        self.battery_proc = BatteryProcessor(self.my_robot_id)

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

        self.timer = self.create_timer(0.1, self.main_controller)

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
            
        elif self.state == RobotState.MOVE_TO_PICKUP:




        with self.lock:
            current_battery = self.battery_percent
            q1 = self.line1_count
            q2 = self.line2_count
            current_status = self.line_status.copy()

        self.battery_proc.pick_up_waiting(
            current_battery,
            q1, 
            q2, 
            current_status
        )

def main(args=None):
    rclpy.init(args=args)
    node = Batterytest()
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