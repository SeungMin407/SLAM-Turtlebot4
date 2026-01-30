import time
import math
import os
import sys
import rclpy
import threading
from queue import Queue, Empty
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


class YOLOImageSubscriber(Node):
    def __init__(self, model):
        super().__init__('yolo_image_subscriber')
        self.model = model
        self.image_queue = Queue(maxsize=1)
        self.should_shutdown = False
        self.classNames = model.names if hasattr(model, 'names') else ['Object']



        # 카메라 토픽 구독
        self.subscription = self.create_subscription(
            CompressedImage,
            '/robot4/oakd/rgb/image_raw/compressed',
            self.rgb_callback,
            10)
        
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()



    def rgb_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if rgb is None or rgb.size == 0:
                return

            if self.image_queue.full():
                try:
                    _ = self.image_queue.get_nowait()
                except Empty:
                    pass
            self.image_queue.put_nowait(rgb) #큐에 넣기

        except Exception as e:
            self.get_logger().error(f"Compressed RGB decode failed: {e}")




    # 추론
    def detection_loop(self):
        while rclpy.ok() and not self.should_shutdown:
            try:
                img = self.image_queue.get(timeout=0.5)  # 큐에 있는걸 가져옴
            except Empty:
                continue
            results = self.model.predict(img)        # 여기서 추론함.
                
            for r in results:                           # 매 결과마다
                if r.boxes is None or len(r.boxes) == 0:        # 탐지 결과가 없다면 스킾
                    continue
                for box in r.boxes:                             # 결과안의 box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])      # yolo에서 제공하는 기능 : 좌표, 클래스, 정확도
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2                         # int는 정수형인데 정확도가 떨어지지 않는가 하는 생각
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) #img에다가 bounding box그림
                    cv2.circle(img, (cx,cy), 4, (0,255,0), -1)
            cv2.imshow("YOLOv8 Detection", img) # 창에 보여주기.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via 'q'")
                self.should_shutdown = True


def main():
    model_name = "best.pt"
    model = YOLO(model_name)
    rclpy.init()
    node = YOLOImageSubscriber(model) #실제 실행

    
    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested via Ctrl+C.")
    finally:
        node.should_shutdown = True
        node.thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
