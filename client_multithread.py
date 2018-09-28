import datetime
from threading import Thread
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import time
import socket
import pickle
import zlib
import struct


class FPS:
    def __init__(self):
        self.start = None
        self.end = None
        self.frameCount = 0

    def start(self):
        self.start = datetime.datetime.now()
        return self

    def stop(self):
        self.end = datetime.datetime.now()

    def updateFrameCount(self):
        self.frameCount += 1

    def fps(self):
        tmp = self.frameCount / (self.end - self.start).total_seconds


class PiVideoStream:
    def __init__(self, resolution=(1280, 720), framerate=30):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format="bgr")
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

while True:
    image = stream.read()
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = zlib.compress(pickle.dumps(frame, 0))
    data = pickle.dumps(frame, 0)
    size = len(data)

    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
