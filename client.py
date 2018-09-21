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
camera = PiCamera()
rawCapture = PiRGBArray(camera)
camera.capture(rawCapture, format="bgr")

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    # ret, frame = cam.read()
    image = rawCapture.array
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)

    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    cv2.imshow('ImageWindow', image)
    img_counter += 1
