from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions, TurtleBot4Navigator
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Quaternion
import math
import time
from builtin_interfaces.msg import Time
class NavProcessor():
    def __init__(self):
        self.navigator = TurtleBot4Navigator()
        #self.navigator.waitUntilNav2Active()

        self.last_detection_time = 0

    def nav_setup(self, start_x, start_y, start_or):
        initial_pose = self.navigator.getPoseStamped([start_x, start_y], start_or)
        self.navigator.setInitialPose(initial_pose)

        self.navigator.waitUntilNav2Active()
        self.navigator.info(f"초기 위치 설정 완료: ({start_x}, {start_y})")

    def go_to_pose(self, goal_x, goal_y, goal_or, start_x=None, start_y=None, start_or=None):
        if start_x is not None and start_y is not None:
            if start_or is None:
                start_or = TurtleBot4Directions.NORTH
            self.nav_setup(start_x, start_y, start_or)
        
        goal_pose = self.navigator.getPoseStamped([goal_x, goal_y], goal_or)
        self.navigator.startToPose(goal_pose)

    def go_to_pose_yaw(self, clock, P_goal, target_yaw):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = clock
        goal_pose.pose.position.x = P_goal[0]
        goal_pose.pose.position.y = P_goal[1]
        goal_pose.pose.position.z = 0.0
        yaw = float(target_yaw)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        self.navigator.goToPose(goal_pose)
    
    def go_to_through(self, goal_array, goal_or, start_x=None, start_y=None, start_or=None):
        if start_x is not None and start_y is not None:
            if start_or is None:
                start_or = TurtleBot4Directions.NORTH
            self.nav_setup(start_x, start_y, start_or)

        goal_pose = []

        for point in goal_array:
            goal_pose.append(self.navigator.getPoseStamped([point[0], point[1]], goal_or))

        self.navigator.startThroughPoses(goal_pose)

    def go_to_follow(self, goal_array, goal_or, start_x=None, start_y=None, start_or=None):
        if start_x is not None and start_y is not None:
            if start_or is None:
                start_or = TurtleBot4Directions.NORTH
            self.nav_setup(start_x, start_y, start_or)

        goal_pose = []

        for point in goal_array:
            goal_pose.append(self.navigator.getPoseStamped([point[0], point[1]], goal_or))

        self.navigator.startFollowWaypoints(goal_pose)

    
    def spin(self, angle_deg):
        angle_rad = math.radians(angle_deg)
        self.navigator.info(f"제자리 돌기{angle_rad}")
        self.navigator.spin(spin_dist=angle_rad, time_allowance=10)
    
    def search_spin_time(self, detections, rotator, undettime, conf_threadhold=0.5):
        curr_time = time.time()

        is_valid_object = False

        if detections:
            for data in detections:
                if data['conf'] >= conf_threadhold:
                    is_valid_object = True
                    break

        if is_valid_object:
            self.last_detection_time = curr_time

            if rotator.is_running:
                print("rotator stop")
                rotator.stop()
            
            return True
        
        time_diff = curr_time - self.last_detection_time

        if time_diff < undettime:
            return True
        elif not rotator.is_running:
                print("rotator start")
                rotator.rotate_step()
        
        return False

    def stop(self):
        self.navigator.info("action stop")
        self.navigator.cancelTask()

    def dock(self):
        if not self.navigator.getDockedStatus():
            print('도킹을 시작합니다...')
            self.navigator.dock()
        else:
            print('이미 도킹되어 있습니다.')

    def undock(self):
        if self.navigator.getDockedStatus():
            print('언독킹을 시작합니다...')
            self.navigator.undock()
        else:
            print('이미 언독되어 있습니다 (로봇 위치 확인 필요).')
    def yaw_to_quat(self, yaw_rad: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw_rad / 2.0)
        q.w = math.cos(yaw_rad / 2.0)
        return q
    def make_pose(self, x, y, yaw_rad):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = Time(sec=0, nanosec=0)
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = self.yaw_to_quat(yaw_rad)
        return pose
    # 회전 없이 깔끔한 waypoint
    def way_point_no_ori(self, goal_array, goal_or=None, start_x=None, start_y=None, start_or=None):
        if start_x is not None and start_y is not None:
            if start_or is None:
                start_or = TurtleBot4Directions.NORTH
            self.nav_setup(start_x, start_y, start_or)

        goal_pose = []
        final_yaw = None
        if goal_or is not None:
            # TurtleBot4Directions 값이 degree이므로 rad로 변환
            final_yaw = math.radians(float(goal_or))
        # 다음 목적지 방향으로 회전
        for i, (x, y) in enumerate(goal_array):
            is_last = (i == len(goal_array) - 1)

            if not is_last:
                nx, ny = goal_array[i + 1]
                yaw = math.atan2(ny - y, nx - x)  # 이동 방향
            else:
                # 최종 방향으로 설정
                if final_yaw is not None:
                    yaw = final_yaw
                else:
                    # goal_or 안 주면 마지막도 "이동방향 유지"(직전 방향) 느낌으로
                    if len(goal_array) >= 2:
                        px, py = goal_array[i - 1]
                        yaw = math.atan2(y - py, x - px)
                    else:
                        yaw = 0.0

            goal_pose.append(self.make_pose(x, y, yaw))

        self.navigator.startFollowWaypoints(goal_pose)
        