import socket
import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import zlib
import datetime
from PyQt5.QtCore import *


class VideoStream(QThread):
    def __init__(self, HOST='73.103.73.130', PORT=8485):
        QThread.__init__(self)
        self.HOST = HOST
        self.PORT = PORT
        self.connection = self.connectServer()
        self.frame = ''

    def connectServer(self):
        """Connects to proxy server

        Returns:
            [socket connection]: connection
        """

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.HOST, self.PORT))
        return s

    def recieveVideo(self):
        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        start_time = datetime.datetime.now()
        img_counter = 0
        while True:
            current_time = datetime.datetime.now()
            if (current_time - start_time).total_seconds() >= 1:
                print("FPS is {}".format(img_counter /
                                         (current_time-start_time).total_seconds()))
                img_counter = 0
                start_time = datetime.datetime.now()
            else:
                while len(data) < payload_size:
                    # print("Recv: {}".format(len(data)))
                    data += self.connection.recv(4096)

            # print("Done Recv: {}".format(len(data)))
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]
                # print("msg_size: {}".format(msg_size))
                while len(data) < msg_size:
                    data += self.connection.recv(4096)
                frame_data = data[:msg_size]
                data = data[msg_size:]
                img_counter = img_counter + 1
                self.frame = pickle.loads(
                    frame_data, fix_imports=True, encoding="bytes")
                self.frame = cv2.imdecode(self.frame, cv2.IMREAD_ANYCOLOR)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_YUV2BGR)
