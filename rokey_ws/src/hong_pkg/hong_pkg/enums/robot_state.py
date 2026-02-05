from enum import Enum, auto

class RobotState(Enum):
    CHARGING = auto()        # 도킹 스테이션에서 대기/충전
    MOVE_TO_PICKUP = auto()  # 라인으로 이동
    WAITTING = auto()        # 배터리 상태와 라인 박스 상태 체크
    LOADING = auto()         # 5초 기다리기 (박스 받는 중)
    MOVE_TO_DEST = auto()    # 목적지로 배달
    RETURN_TO_LINE = auto()  # 라인으로 복귀 (다음 작업 대기)
    STOP = auto()
    DOCKING = auto()