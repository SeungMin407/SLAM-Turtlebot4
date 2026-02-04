import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Int32
from enum import Enum, auto
from threading import Lock


class RobotState(Enum):
    START = auto()
    ROBOT_READY = auto()
    SEARCHING = auto()
    WAITING_USER = auto()
    APPROACHING = auto()
    DONE = auto()


class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')

        self.lock = Lock()
        self.state = RobotState.START

        self.qr_topic1 = '/qr_code_id1'
        self.qr_topic2 = '/qr_code_id2'
        self.qr_subscriber1 = self.create_subscription(
            Int32,
            self.qr_topic1,
            self.qr_callback1,
            qos_profile_sensor_data
        )
        self.qr_subscriber2 = self.create_subscription(
            Int32,
            self.qr_topic2,
            self.qr_callback2,
            qos_profile_sensor_data
        )
    def qr_callback1(self, msg):
        with self.lock:
            self.get_logger().info(f"QR1 received: {msg.data}")
            self.state = RobotState.ROBOT_READY
    def qr_callback2(self, msg):
        with self.lock:
            self.get_logger().info(f"QR2 received: {msg.data}")
            self.state = RobotState.ROBOT_READY


def main():
    rclpy.init()
    node = TestNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
