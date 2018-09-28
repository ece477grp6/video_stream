import datetime
from threading import Thread
import cv2
from picamera.array import PiRGBArray
from picamera.array import PiYUVArray
from picamera import PiCamera
import argparse
import time
import socket
import pickle
import zlib
import struct


class PiVideoStream:
    def __init__(self, resolution=(720, 480), framerate=24):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiYUVArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format="yuv")
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
        if self.stopped:
            self.stream.close()
            self.rawCapture.close()
            self.camera.close()
            return

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
stream = PiVideoStream().start()
time.sleep(1)
start_time = datetime.datetime.now()
while True:
    # current_time = datetime.datetime.now()
    # if (current_time - start_time).total_seconds() >= 1:
    #     print("FPS is {}".format(img_counter /
    #                              (current_time-start_time).total_seconds()))
    #     img_counter = 0
    #     start_time = datetime.datetime.now()
    # else:
    image = stream.read()
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)

    # print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
