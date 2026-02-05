from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

class DriveCommander:
    def __init__(self, node):
        self.node = node
        self.publisher = node.create_publisher(Twist, '/robot4/cmd_vel', 10)
        self.publisher = node.create_publisher(Int32MultiArray,'/line_status',10)

    def send_velocity(self, linear_x, angular_z):
        msg = Twist()
        
        msg.linear.x = float(linear_x)
        msg.linear.y = 0.0              
        msg.linear.z = 0.0              

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)

        self.publisher.publish(msg)
    
    def publish_line_status(self, is_line1_busy, is_line2_busy):
        msg = Int32MultiArray()

        status_data = [0, 0, 0]

        if is_line1_busy:
            status_data[1] = 1
            
        if is_line2_busy:
            status_data[2] = 1
            
        msg.data = status_data
        
        self.status_publisher.publish(msg)


    def stop(self):
        self.send_velocity(0.0, 0.0)