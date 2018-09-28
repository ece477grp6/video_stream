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
    fps.update()

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
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
