from irobot_create_msgs.action import RotateAngle
from action_base import ActionBase 

class RotationManager(ActionBase):
    def __init__(self, node):
        super().__init__(node, RotateAngle, '/rotate_angle')

    def rotate_step(self):
        goal = RotateAngle.Goal()
        goal.angle = 0.785
        goal.max_rotation_speed = 0.3
        
        self.node.get_logger().info("탐지된 객체 없음 -> 회전 수행")

        self.send_goal_base(goal)

    def stop(self):
        self.node.get_logger().info("객체 발견! -> 회전 중지")
        self.cancel_base()