import cv2
import io
import socket
import struct
import time
import pickle
import zlib
from picamera.array import PiRGBArray
from picamera import PiCamera

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
with PiCamera() as camera:
    camera.resolution = (1280, 720)
    camera.framerate = 24
    rawCapture = PiRGBArray(camera, size=(1280, 720))
    time.sleep(0.1)

    for camera.capture_continuous(rawCapture, format="bgr"):
        image = rawCapture.array
        result, frame = cv2.imencode('.jpg', image, encode_param)
        data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)

        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
        img_counter += 1
        rawCapture.truncate(0)
