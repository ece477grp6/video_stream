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

    def run(self):
        self.portLock = threading.Lock()
        self.portLock.acquire()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # logging.info("Socket created for " + self.portname)
        print("Socket created for " + self.portname))
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
        self.portLock.release()


class TextServer(TcpServer):
    data = b''
    lockRecv = threading.Lock()
    lockSend = threading.Lock()

    def recvData(self):
        # while True:
        #     try:
        #         self.conn
        #     except:
        #         pass
        #     else:
        #         logging.info(self._name + " Client connected from " +
        #                      self.addr[0] + ":" + str(self.addr[1]))
        #         break
        self.portLock.acquire
        try:
            while True:
                self.lockSend.acquire()
                try:
                    self.lockRecv.acquire()
                    try:
                        self.data = self.conn.recv(1024)
                    except:
                        continue
                    else:
                        # logging.info("Received text data")
                        print("Received text data")
                        self.lockRecv.release()
                        print("RELEASE")
                except:
                    continue
                else:
                    pass
        except:
            pass
        else:
            self.portLock.release()

    def sendData(self):
        # while True:
        #     try:
        #         self.conn
        #     except:
        #         pass
        #     else:
        #         logging.info("Client connected from " +
        #                      self.addr[0] + " on port " + str(self.addr[1]))
        #         break
        self.portLock.acquire()
        try:
            while True:
                self.lockRecv.acquire()
                try:
                    # self.lockSend.acquire()
                    self.conn.sendall(self.data)
                    # logging.info("Sent text data")
                    print("Sent text data"))
                except:
                    continue
                else:
                    self.lockSend.release()
                    self.lockRecv.release()
        except:
            pass
        else:
            self.portLock.release()


class VideoServer(TcpServer):
    data = b''
    lock = threading.Lock()

    def recvData(self):
        while True:
            try:
                self.conn
            except:
                pass
            else:
                # logging.info("Client connected from " +
                #              self.addr[0] + " on port " + str(self.addr[1]))
                print("Client connected from " +
                             self.addr[0] + " on port " + str(self.addr[1]))
                break
        while True:
            self.lock.acquire()
            try:
                self.data = self.conn.recv(4096)
                # logging.info("Received video data")
                print("Received video data")
            finally:
                self.lock.release()

    def sendData(self):
        while True:
            try:
                self.conn
            except:
                pass
            else:
                # logging.info("Client connected from " +
                #              self.addr[0] + " on port " + str(self.addr[1]))
                print("Client connected from " +
                             self.addr[0] + " on port " + str(self.addr[1]))
                break
        while True:
            self.lock.acquire()
            try:
                self.conn.sendall(self.data)
                # logging.info("Sent text data")
                print("Sent text data")
            finally:
                self.lock.release()


class StartServer():
    def __init__(self, recvServer, sendServer):
        # recvServer.run()
        # sendServer.run()
        # print(recvServer.conn)
        recvThread = threading.Thread(target=recvServer.recvData)
        sendThread = threading.Thread(target=sendServer.sendData)
        # recvThread.daemon = True
        # sendThread.daemon = True
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
    textRecv.daemon = True
    textSend.daemon = True
    videoRecv.daemon = True
    videoSend.daemon = True
    textRecv.start()
    textSend.start()
    videoRecv.start()
    videoSend.start()
    # while 1:
    #     pass
    textServer = StartServer(textRecv, textSend)
    # videoServer = StartServer(videoRecv, videoSend)


if __name__ == '__main__':
    main()
