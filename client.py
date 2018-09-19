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
client_socket.connect(('10.186.44.62', 8485))
connection = client_socket.makefile('wb')
camera = PiCamera()
rawCapture = PiRGBArray(camera)
time.sleep(0.1)
# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array

# cam = cv2.VideoCapture(rawCapture)

# cam.set(3, 320)
# cam.set(4, 240)

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', image, encode_param)
#    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)

    # print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    # img_counter += 1

cam.release()
