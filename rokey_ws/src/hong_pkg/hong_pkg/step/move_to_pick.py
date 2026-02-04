#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from enum import Enum, auto
from threading import Lock

from .utils.nav_util import NavProcessor
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions


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
        self.nav = NavProcessor()
        self.state = RobotState.ROBOT_READY

        # pick place 좌표(수정)
        self.ready_goal = (-12.0, 0.0, TurtleBot4Directions.NORTH)

        self.get_logger().info("Waiting for state == ROBOT_READY")
        self.timer = self.create_timer(0.1, self.go_pick) #토픽 안받고 로봇 상태로 실행-> 타이머 생성

    def go_pick(self):
        with self.lock:
            #  ROBOT_READY면 이동
            if self.state == RobotState.ROBOT_READY:
                self.state=RobotState.APPROACHING
                x, y, d = self.ready_goal
                self.get_logger().info(
                    f"State is ROBOT_READY → moving to ({x}, {y})"
                )
                self.nav.go_to_pose(
                    goal_x=x,
                    goal_y=y,
                    goal_or=d
                )
                self.state = RobotState.DONE #이동 끝나면 DONE으로 상태 변경
                self.get_logger("DONE")

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
