#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration


AUDIO_TOPIC = '/robot4/cmd_audio'   


class BeepNode(Node):
    def __init__(self):
        super().__init__('beep_node')

 
        self.pub = self.create_publisher(
            AudioNoteVector,
            AUDIO_TOPIC,
            10,   
        )


        self.timer = self.create_timer(0.5, self.beep_once)
        self.already_sent = False

    def make_note(self, freq_hz: float, duration_sec: float) -> AudioNote:
        note = AudioNote()


        note.frequency = int(freq_hz)

        sec = 0
        nsec = int(duration_sec * 1e9)
        note.max_runtime = Duration(sec=sec, nanosec=nsec)

        return note

    def beep_once(self):
        if self.already_sent:
            return

        msg = AudioNoteVector()


        msg.append = False

        # CLI와 동일한 패턴: 삐(880) 뽀(440) 삐(880) 뽀(440)
        msg.notes = [
            self.make_note(880.0, 0.3),
            self.make_note(440.0, 0.3),
            self.make_note(880.0, 0.3),
            self.make_note(440.0, 0.3),
        ]

        self.pub.publish(msg)
        self.get_logger().info('삐뽀삐뽀 경고음 전송!')

        self.already_sent = True
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = BeepNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
