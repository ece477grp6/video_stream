import sys
import socket
import os
import logging
import threading
HOST = ''
VIDEO_IN_PORT = 8585
VIDEO_OUT_PORT = 8586
TEXT_IN_PORT = 8686
TEXT_OUT_PORT = 8687


class TcpServer(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        if (port == 8585):
            portname = "video in"
        elif (port == 8586):
            portname = "video out"
        elif port == 8686:
            portname = "text in"
        else:
            portname = "text out"
        self.portname = portname
        self.port = port

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info("Socket created for " + self.portname)
        self.s.bind(HOST, self.port)
        logging.info('Socket bind complete for ' + self.portname)
        self.s.listen(10)
        logging.info('Socket now listening for ' + self.portname)
        self.conn, self.addr = self.s.accept()


class TextServer(TcpServer):
    data = b''
    lock = threading.Lock()

    def recvData(self):
        self.lock.acquire()
        try:
            self.data = self.s.recv(1024)
            logging.info("Received text data")
        finally:
            self.lock.release()

    def sendData(self):
        self.lock.acquire()
        try:
            self.s.sendall(self.data)
            logging.info("Sent text data")
        finally:
            self.lock.release()


class VideoServer(TcpServer):
    data = b''
    lock = threading.Lock()

    def recvData(self):
        self.lock.acquire()
        try:
            self.data = self.s.recv(4096)
            logging.info("Received video data")
        finally:
            self.lock.release()

    def sendData(self):
        self.lock.acquire()
        try:
            self.s.sendall(self.data)
            logging.info("Sent text data")
        finally:
            self.lock.release()


class StartServer():
    def __init__(self, recvServer, sendServer):
        recvServer.run()
        sendServer.run()
        recvThread = threading.Thread(target=recvServer.recvData())
        sendThread = threading.Thread(target=sendServer.sendData())
        recvThread.start()
        sendThread.start()


def main():
    if os.path.exists("Server.log"):
        os.remove("Server.log")
    logging.basicConfig(filename='Server.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    textRecv = TextServer(TEXT_IN_PORT)
    textSend = TextServer(TEXT_OUT_PORT)
    videoRecv = VideoServer(VIDEO_IN_PORT)
    videoSend = VideoServer(VIDEO_OUT_PORT)
    textRecv.start()
    textSend.start()
    videoRecv.start()
    videoSend.start()
    textServer = StartServer(textRecv, textSend)
    videoServer = StartServer(videoRecv, videoSend)


if __name__ == '__main__':
    main()
