import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, PoseWithCovarianceStamped

qos_amcl = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)


class NodeTest(Node):
    def __init__(self):
        super().__init__('node_test')
        self.robot_x = None
        self.robot_y = None
        self.create_subscription(PoseWithCovarianceStamped, '/robot4/amcl_pose', self.amcl_callback, qos_amcl)
        print("안녕하세요")

    def amcl_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        self.robot_x = x
        self.robot_y = y

        self.get_logger().info(f'robot x = {self.robot_x}, y = {self.robot_y}')

def main():
    rclpy.init()
    node = NodeTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()