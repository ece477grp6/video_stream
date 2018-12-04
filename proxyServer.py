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


class TextServer():
    def startForward(self, recvServer, sendServer):
        # delayFlag = True
        while True:
            if recvServer.connected and sendServer.connected:
                try:
                    self.data = recvServer.conn.recv(1024)
                    # logging.info("Recieved text data " + str(self.data))
                    print("Recieved text data " + str(self.data))
                    # recvServer.conn.sendall(b'OK')
                    # print("SEND OK")
                    time.sleep(1)
                    sendServer.conn.sendall(self.data)
                    # logging.info("Text data sent " + str(self.data))
                    print("Text data sent " + str(self.data))
                    self.data = sendServer.conn.recv(1024)
                    # logging.info("Recieved reply " + str(self.data))
                    print("Recieved reply " + str(self.data))
                    recvServer.conn.sendall(self.data)
                    # logging.info("Text reply sent " + str(self.data))
                    print("Text reply sent " + str(self.data))
                except Exception as e:
                    # logging.info(e)
                    print(str(e))


class VideoServer():
    def startForward(self, recvServer, sendServer):
        while True:
            if recvServer.connected and sendServer.connected:
                try:
                    self.data = recvServer.conn.recv(4096)
                    # logging.info("Recieved text data " + str(self.data))
                    print("Recieved text data " + str(self.data))
                    sendServer.conn.sendall(self.data)
                    # logging.info("Text data sent " + str(self.data))
                    print("Text data sent " + str(self.data))
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
    textServer = threading.Thread(
        target=TextServer().startForward(textRecv, textSend))
    videoServer = threading.Thread(
        target=TextServer().startForward(videoRecv, videoSend))
    textServer.daemon = True
    videoServer.daemon = True
    textServer.start()
    videoServer.start()


if __name__ == "__main__":
    main()
