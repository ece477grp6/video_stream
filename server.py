import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import zlib
import datetime
HOST = ''
PORT = 8485

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

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
            data += conn.recv(4096)

    # print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        # print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        img_counter = img_counter + 1
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        # frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_Y422)
        frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)
        cv2.imshow('ImageWindow', frame)
        cv2.waitKey(1)
