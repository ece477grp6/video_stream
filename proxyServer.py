import sys
import socket
import os
import logging
import threading
import time
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
        self.connected = False

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # logging.info("Socket created for " + self.portname)
        print("Socket created for " + self.portname)
        self.s.bind((HOST, self.port))
        # logging.info('Socket bind complete for ' + self.portname)
        print('Socket bind complete for ' + self.portname)
        self.s.listen(10)
        # logging.info('Socket now listening for ' + self.portname)
        print('Socket now listening for ' + self.portname)
        self.conn, self.addr = self.s.accept()
        # logging.info(self._name + " Client connected from " +
        #              self.addr[0] + ":" + str(self.addr[1]))
        print(self._name + " Client connected from " +
              self.addr[0] + ":" + str(self.addr[1]))
        self.connected = True


class TextServer(threading.Thread):
    def __init__(self, recvServer, sendServer):
        self.recvServer = recvServer
        self.sendServer = sendServer
        threading.Thread.__init__(self)

    def run(self):
        # delayFlag = True
        while True:
            if self.recvServer.connected and self.sendServer.connected:
                try:
                    self.data = self.recvServer.conn.recv(1024)
                    # logging.info("Recieved text data " + str(self.data))
                    print("Recieved text data " + str(self.data))
                    # recvServer.conn.sendall(b'OK')
                    # print("SEND OK")
                    time.sleep(1)
                    self.sendServer.conn.sendall(self.data)
                    # logging.info("Text data sent " + str(self.data))
                    print("Text data sent " + str(self.data))
                    self.data = self.sendServer.conn.recv(1024)
                    # logging.info("Recieved reply " + str(self.data))
                    print("Recieved reply " + str(self.data))
                    self.recvServer.conn.sendall(self.data)
                    # logging.info("Text reply sent " + str(self.data))
                    print("Text reply sent " + str(self.data))
                except Exception as e:
                    # logging.info(e)
                    print(str(e))


class VideoServer(threading.Thread):
    def __init__(self, recvServer, sendServer):
        threading.Thread.__init__(self)
        self.recvServer = recvServer
        self.sendServer = sendServer
    def run(self):
        while True:
            if self.recvServer.connected and self.sendServer.connected:
                try:
                    # print("RECV")
                    self.data = self.recvServer.conn.recv(4096)
                    # logging.info("Recieved text data " + str(self.data))
                    print("Recieved video data ")
                    self.sendServer.conn.sendall(self.data)
                    # logging.info("Text data sent " + str(self.data))
                    print("Video data sent ")
                except Exception as e:
                    # logging.info(e)
                    print(str(e))


def main():
    logging.basicConfig(filename='Server.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
    textRecv = TcpServer(TEXT_IN_PORT)
    textSend = TcpServer(TEXT_OUT_PORT)
    videoRecv = TcpServer(VIDEO_IN_PORT)
    videoSend = TcpServer(VIDEO_OUT_PORT)
    textRecv.daemon = True
    textSend.daemon = True
    videoRecv.daemon = True
    videoSend.daemon = True
    textRecv.start()
    time.sleep(0.1)
    textSend.start()
    time.sleep(0.1)
    videoRecv.start()
    time.sleep(0.1)
    videoSend.start()
    # textServer = threading.Thread(
    #     target=TextServer().startForward(textRecv, textSend))
    # videoServer = threading.Thread(
    #     target=VideoServer().startForward(videoRecv, videoSend))
    videoServer = VideoServer(videoRecv, videoSend)
    textServer = TextServer(textRecv, textSend)
    textServer.daemon = True
    videoServer.daemon = True
    videoServer.start()
    textServer.start()
    while True:
        pass


if __name__ == "__main__":
    main()
