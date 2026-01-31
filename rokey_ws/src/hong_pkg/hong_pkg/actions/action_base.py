from rclpy.action import ActionClient

class ActionBase:
    def __init__(self, node, action_type, action_topic):
        self.node = node
        self._client = ActionClient(node, action_type, action_topic)
        self._handle = None
        self.is_running = False  # ★ 현재 액션이 실행 중인지 확인하는 플래그

    def send_goal_base(self, goal_msg):
        self._client.wait_for_server()
        future = self._client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)
        self.is_running = True  # 실행 시작 표시

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.is_running = False
            return
        
        self._handle = goal_handle
        get_result = goal_handle.get_result_async()
        get_result.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        """공통: 액션 종료 시 (성공이든 취소든)"""
        self.is_running = False
        self._handle = None

    def cancel_base(self):
        """공통: 취소"""
        if self._handle:
            self._handle.cancel_goal_async()
            self.is_running = False