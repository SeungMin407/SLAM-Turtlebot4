import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rokey/turtlebot4_ws/install/turtlebot4_openai_tutorials'
