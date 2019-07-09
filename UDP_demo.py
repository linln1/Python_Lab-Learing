# usr/bin/env python3
# -*- encoding:utf-8 -*-

# 服务器
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind('127.0.0.1', 9999)
print('Bind UDP on 9999...')
while True:
     data ,addr = s.recvfrom(1024)
     print('Received from $s:%s.' % addr)
     s.sendto(b'Hello, %s!' % data,addr)
     
#客户端
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for data in [b'Michael', b'Tracy', b'Sarah']:
      s.sendto(data, ('127.0.0.1', 9999)
      print(s.recv(1024).decode('utf-8'))
s.close()
