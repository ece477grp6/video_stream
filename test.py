import numpy as np
import cv2
import socket
import datetime
import zlib
import pickle
import struct

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.0.1.190', 8585))
connection = client_socket.makefile('wb')
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
stream = cv2.VideoCapture(0)
start_time = datetime.datetime.now()

while True:
    # current_time = datetime.datetime.now()
    # if (current_time - start_time).total_seconds() >= 1:
    #     print("FPS is {}".format(img_counter /
    #                              (current_time-start_time).total_seconds()))
    #     img_counter = 0
    #     start_time = datetime.datetime.now()
    # else:
    frame = stream.read()
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)

    # print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1

# When everything done, release the capture
cap.release()
