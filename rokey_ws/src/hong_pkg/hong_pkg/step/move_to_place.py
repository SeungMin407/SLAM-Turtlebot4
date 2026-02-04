import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Bool, Int32
from enum import Enum, auto
from threading import Lock
from collections import deque

from .utils.nav_util import NavProcessor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions


class RobotState(Enum):
    START = auto()
    ROBOT_READY = auto()
    SEARCHING = auto()
    WAITING_USER = auto()
    APPROACHING = auto()
    DONE = auto()

#qr class에 따라 다른 좌표(place 위치)로 이동
class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')

        self.lock = Lock()
        self.state = RobotState.START
        self.nav = NavProcessor()

        # topics
        self.box_placed_topic = '/box_placed'
        self.qr_id1_topic = '/qr_code_id1'

        self.box_placed_sub = self.create_subscription(
            Bool,
            self.box_placed_topic,
            self.box_placed_callback,
            qos_profile_sensor_data
        )

        self.qr_id1_sub = self.create_subscription(
            Int32,
            self.qr_id1_topic,
            self.qr_id1_callback,
            qos_profile_sensor_data
        )

        # QR 클래스에 따른 목표 좌표(조정 필요)
        self.qr_goal_map = {
            1: (-14.0, 0.0, TurtleBot4Directions.SOUTH),
            2: (-14.5, 1.0, TurtleBot4Directions.SOUTH),
            3: (-14.0, 1.5, TurtleBot4Directions.SOUTH),
        }

        # QR queue (FIFO)
        self.qr_queue = deque(maxlen=10)

        self.get_logger().info("AMR ready: queueing /qr_code_id1, consuming on /box_placed")

    def move_to_goal(self, goal_x: float, goal_y: float, goal_or):
        # 이미 이동 중이면 중복 트리거 방지
        if self.state == RobotState.APPROACHING:
            self.get_logger().warn("Already APPROACHING. Cannot start new goal now.")
            return False

        self.state = RobotState.APPROACHING
        self.get_logger().info(f"Moving to goal: x={goal_x}, y={goal_y}")

        self.nav.go_to_pose(goal_x=goal_x, goal_y=goal_y, goal_or=goal_or)
        return True

    # Callbacks
    def qr_id1_callback(self, msg: Int32):
        with self.lock:
            qr_id = int(msg.data)
            self.qr_queue.append(qr_id) #qr 오면 큐에 넣어놓고 순차적 작업
            self.get_logger().info(
                f"/qr_code_id1 received: {qr_id} -> queued (size={len(self.qr_queue)})"
            )

    def box_placed_callback(self, msg: Bool):
        with self.lock:
            self.get_logger().info(f"/box_placed received: {msg.data}")

            if not msg.data:
                return

            # 이동 중이면 이번 트리거는 무시 (큐는 계속 쌓임)
            if self.state == RobotState.APPROACHING:
                self.get_logger().warn(
                    f"Box placed but robot is APPROACHING. Queue size={len(self.qr_queue)}"
                )
                return

            # 큐가 비었으면 할 일이 없음
            if len(self.qr_queue) == 0:
                self.get_logger().warn("Box placed but QR queue is empty. Nothing to do.")
                return

            # 큐에서 하나 꺼내서 목적지 결정
            qr_id = self.qr_queue.popleft()
            self.get_logger().info(f"Dequeue qr_id={qr_id} (remaining={len(self.qr_queue)})")

            if qr_id not in self.qr_goal_map:
                self.get_logger().warn(f"Unknown qr_id={qr_id}. Skip.")
                return

            goal_x, goal_y, goal_or = self.qr_goal_map[qr_id]
            self.move_to_goal(goal_x, goal_y, goal_or)


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


if __name__ == "__main__":
    main()
