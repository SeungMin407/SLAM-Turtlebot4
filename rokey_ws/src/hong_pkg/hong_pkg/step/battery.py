import json
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import BatteryState
from std_msgs.msg import String


class BatteryAggregator(Node):
    def __init__(self):
        super().__init__('battery_aggregator')

        self.robot_ids = [4, 5]
        self.latest = {4: None, 5: None}  # 0.0~1.0

        # subs
        self.sub4 = self.create_subscription(
            BatteryState, '/robot4/battery_state',
            lambda msg: self.on_battery(4, msg),
            10
        )
        self.sub5 = self.create_subscription(
            BatteryState, '/robot5/battery_state',
            lambda msg: self.on_battery(5, msg),
            10
        )

        # pub
        self.pub = self.create_publisher(String, '/battery', 10)

        # publish rate
        self.timer = self.create_timer(0.1, self.publish_all)

        self.get_logger().info('Sub: /robot4/battery_state, /robot5/battery_state')
        self.get_logger().info('Pub: /robots/battery_percentages')

    def on_battery(self, robot_id: int, msg: BatteryState):
        if msg.percentage is None:
            return
        self.latest[robot_id] = float(msg.percentage)

    def publish_all(self):
        # 로봇 번호, 배터리 퍼센트 퍼블리시, {"4": 0.91, "5": 0.92} 형태
        payload = {str(k): self.latest[k] for k in self.robot_ids}
        self.pub.publish(String(data=json.dumps(payload)))


def main():
    rclpy.init()
    node = BatteryAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
