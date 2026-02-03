import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState

# threading 모듈 수정 (lock -> Lock)
from threading import Lock, Thread
from enum import Enum, auto

# custom class
from .modules.cooperation_process import CooperationProcess

class RobotState(Enum):
    START = auto()
    ROBOT_READY = auto()
    SEARCHING = auto()
    WAITING_USER = auto()
    APPROACHING = auto()
    DONE = auto()

class Batterytest(Node):
    def __init__(self):
        super().__init__('battery_test_node') # 노드 이름 명확하게 변경

        # [중요] Lock 객체 생성 (괄호 필수)
        self.lock = Lock()

        # 변수 초기화
        self.state = RobotState.START
        self.battery_percent = 0.0 # None 보다는 float 초기값 권장

        self.battery = CooperationProcess()

        # Depth Topic 변수는 현재 안 쓰이지만 일단 유지
        ns = self.get_namespace()
        self.depth_topic = f'{ns}/oakd/stereo/image_raw'

        # 배터리 구독 설정
        self.battery_state_subscriber = self.create_subscription(
            BatteryState,
            'battery_state', # 토픽 이름 (필요 시 /battery_state 로 변경)
            self.battery_state_callback,
            qos_profile_sensor_data
        )

    def battery_state_callback(self, batt_msg: BatteryState):
        # 스레드 락 사용
        with self.lock:
            self.battery_percent = batt_msg.percentage

    def rootController(self):
        if self.state == RobotState.START:
            self.battery.battery_check(self.battery_percent)


def main(args=None):
    rclpy.init(args=args)
    
    node = Batterytest()
    
    # 멀티 스레드 실행기 사용
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