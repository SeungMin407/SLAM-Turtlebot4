from irobot_create_msgs.action import DriveDistance
from .action_base import ActionBase 


class DriveManager(ActionBase):
    def __init__(self, node):
        super().__init__(node, DriveDistance, '/robot4/drive_distance')
        
    
    def drive_step(self, dic=0.1):
        goal = DriveDistance.Goal()
        goal.distance = dic
        goal.max_translation_speed = 0.3

        self.send_goal_base(goal)

    def stop(self):

        self.cancel_base()
        
