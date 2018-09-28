import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from picamera.array import PiRGBArray
from picamera import PiCamera
import datetime
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

# while True:
#     # ret, frame = cam.read()
#     camera.capture(rawCapture, format="bgr")
#     image = rawCapture.array
#     result, frame = cv2.imencode('.jpg', image, encode_param)
#     data = zlib.compress(pickle.dumps(frame, 0))
#     data = pickle.dumps(frame, 0)
#     size = len(data)

#     print("{}: {}".format(img_counter, size))
#     client_socket.sendall(struct.pack(">L", size) + data)
#     img_counter += 1
start_time = datetime.datetime.now()
with PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(1280, 720))
    time.sleep(0.1)

    for foo in camera.capture_continuous(rawCapture, format="bgr"):
        current_time = datetime.datetime.now()
        if (current_time - start_time).total_seconds() >= 1:
            print("FPS is {}".format(img_counter /
                                     (current_time-start_time).total_seconds()))
            img_counter = 0
            start_time = datetime.datetime.now()
        else:
            image = rawCapture.array
            result, frame = cv2.imencode('.jpg', image, encode_param)
            data = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame, 0)
            size = len(data)

            print("{}: {}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1
            rawCapture.truncate(0)
