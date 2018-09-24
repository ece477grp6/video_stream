# import cv2
# import io
# import socket
# import struct
# import time
# import pickle
# import zlib
# from picamera.array import PiRGBArray
# from picamera import PiCamera

# # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # client_socket.connect(('127.0.0.1', 8485))
# # connection = client_socket.makefile('wb')
# img_counter = 0
# encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
# with PiCamera() as camera:
#     camera.resolution = (1280, 720)
#     camera.framerate = 24
#     rawCapture = PiRGBArray(camera, size=(1280, 720))
#     time.sleep(0.1)

#     for foo in camera.capture_continuous(rawCapture, format="bgr"):
#         image = rawCapture.array
#         result, frame = cv2.imencode('.jpg', image, encode_param)
#         data = zlib.compress(pickle.dumps(frame, 0))
#         data = pickle.dumps(frame, 0)
#         size = len(data)

#         print("{}: {}".format(img_counter, size))
#         # client_socket.sendall(struct.pack(">L", size) + data)
#         img_counter += 1
#         rawCapture.truncate(0)
#         cv2.imshow('ImageWindow', frame)
#         cv2.waitKey(1)
import datetime
from threading import Thread
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import time


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


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
        Thread(target=self.upgrade, args=()).start()
        return self

    def update(self):
        for frame in self.stream:
            self.frame = frame.array
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


arg = argparse.ArgumentParser()
arg.add_argument("-n", "--num-frames", type=int, default=100,
                 help="# of frames to loop over for FPS test")
arg.add_argument("-d", "--display", type=int, default=-1,
                 help="Whether or not frames should be displayed")
args = vars(arg.parse_args())

camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=camera.resolution)
stream = camera.capture_continuous(rawCapture, format="bgr")
time.sleep(1)
fps = FPS().start()
print("[INFO] sampling frames from `picamera` module...")
# loop over some frames
for (i, f) in enumerate(stream):
    # grab the frame from the stream and resize it to have a maximum
    # width of 400 pixels
    frame = f.array

    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame and update
    # the FPS counter
    rawCapture.truncate(0)
    fps.updateFrameCount()

    # check to see if the desired number of frames have been reached
    if i == args["num_frames"]:
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
stream.close()
rawCapture.close()
camera.close()
# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
    frame = vs.read()

    # check to see if the frame should be displayed to our screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.updateFrameCount()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
