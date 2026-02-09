from ..utils.nav_util import NavProcessor
from ..enums.robot_state import RobotState
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Directions
import time

ROBOT_CONFIG = {
    4: {  # 로봇 4번 설정
        'my_line': 1,
        'other_line': 2,
        'dock_coords': [[-1.59, -0.47]],    # 도킹 대기 장소
        'support_coords': [[-2.11,-1.55], [-2.87, -1.66]]  # 2번 라인 지원 장소
    },
    'default': {  # 그 외 로봇 (예: 5번) 설정
        'my_line': 2,
        'other_line': 1,
        'dock_coords': [[-1.53, 0.85]],     # 도킹 대기 장소
        'support_coords': [[-2.11,-1.55], [-1.61, -1.70]]  # 1번 라인 지원 장소
    }
}

class MainProcessor:
    def __init__(self, my_robot_id):
        self.robot_id = my_robot_id
        self.nav = NavProcessor()

        # 로봇 ID에 맞는 설정을 불러옵니다. (없으면 default 사용)
        config = ROBOT_CONFIG.get(self.robot_id, ROBOT_CONFIG['default'])
        
        self.my_line_id = config['my_line']
        self.other_line_id = config['other_line']
        self.dock_coords = config['dock_coords']
        self.support_coords = config['support_coords']

        print(f"🤖 Robot {self.robot_id} 초기화 완료 (My Line: {self.my_line_id})")

    def pick_up_waiting(self, battery_percent, my_queue_count, other_queue_count, line_status, my_start):
        battery = battery_percent * 100 if battery_percent <= 1.0 else battery_percent

        if battery < 30:
            print(f'⚡ 배터리 부족({battery:.1f}%)! 도킹 장소로 이동합니다.')
            self.move_and_wait(self.dock_coords, None)
            return RobotState.DOCKING

        elif my_queue_count > 0:
            if line_status.get(self.my_line_id) == True:
                print(f"✋ 내 라인({self.my_line_id}) 작업 대기 중 (Occupied)...")
                if my_start == True:
                    return RobotState.LOADING
                return RobotState.WAITTING
            return RobotState.LOADING

        elif other_queue_count > 0:
            if line_status.get(self.other_line_id) == True:
                print(f"✋ {self.other_line_id}번 라인 지원 대기 중 (Occupied)...")
                if my_start == True:
                    return RobotState.LOADING
                return RobotState.WAITTING
            
            time.sleep(2.0)
            self.move_and_wait(self.support_coords, TurtleBot4Directions.EAST)
            return RobotState.GO_TO_OTHER
        else:
            return RobotState.WAITTING

    def move_and_wait(self, goal_array, goal_or):
        self.nav.way_point_no_ori(goal_array=goal_array, goal_or=goal_or)
        
        while not self.nav.navigator.isTaskComplete():
            time.sleep(0.1)

        print("✅ 도착 완료 (Action Complete)")