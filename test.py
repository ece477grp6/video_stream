import sys
import socket
import select

HOST = ''
SOCKET_LIST = []
RECV_BUFFER = 4096
PORT = 8485


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
print('Socket bind complete for ')
s.listen(10)
print('Socket now listening for ')
conn, addr = s.accept()
print(conn)
print(addr)
print(addr[1a])
