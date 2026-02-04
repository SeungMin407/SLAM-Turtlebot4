from ..utils.nav_util import NavProcessor
import time

class BatteryProcessor:
    def __init__(self, my_robot_id):

        # 로봇 아이디별로 라인 배정
        if my_robot_id == 4:
            self.my_line_id = 1
            self.other_line_id = 2
        else :
            self.my_line_id = 2
            self.other_line_id = 1
        self.nav = NavProcessor()

    # battery percent, 자신의 라인 박스 갯수, 다른 라인 박스 갯수, 각 라인 작업 상태
    def pick_up_waiting(self, battery_percent, my_queue_count, other_queue_count, line_status):
        battery = battery_percent * 100
        # 배터리가 30프로 미만일때 도킹하러감
        if battery < 30:
            print('Low Battery! Go to Dock')
            self.move_and_wait(-11.97, 0.72, 0.0)
            return
        # 자신의 라인의 박스 갯수가 있으면 우선적으로 처리
        elif my_queue_count > 0:
            if line_status.get(self.my_line_id) == True:
                print(f"내 라인({self.my_line_id}) 작업 대기 중 (Occupied)...")
                return

            self.move_and_wait(-12.14, -0.37, 0.0)
            print(f'내 라인({self.my_line_id}) 작업 시작')
        # 자신의 라인에 박스가 없으면 상대 라인 박스 협동
        elif other_queue_count > 0:
            if line_status.get(self.other_line_id) == True:
                print(f"{self.other_line_id}번 지원 대기 중 (Occupied)...")
                return

            self.move_and_wait(-12.21, -0.37, 0.0)
            print(f"{self.other_line_id}번 라인 지원 출발")
        else:
            pass
    # x 좌표, y 좌표, 로봇이 바라보는 방향
    def move_and_wait(self, x, y, yaw):
        self.nav.go_to_pose(x, y, yaw)
        # waitting
        while not self.nav.navigator.isTaskComplete():
            time.sleep(0.1)

        print("도착 완료 (Action Complete)")