from irobot_create_msgs.action import DriveDistance
from .action_base import ActionBase 


class DriveDisManager(ActionBase):
    def __init__(self, node):
        super().__init__(node, DriveDistance, '/robot4/drive_distance')
        
    
    def drive_step(self, dic=0.1, speed=0.3):
        goal = DriveDistance.Goal()
        goal.distance = dic
        goal.max_translation_speed = speed
        self.node.get_logger().info("물체에 다가갑니다")
        self.send_goal_base(goal)

    def stop(self):
        self.node.get_logger().info("종료합니다.")
        self.cancel_base()
        
