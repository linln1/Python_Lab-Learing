# 客户端
# usr/bin/env python3
# -*- coding: utf-8 -*-

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 9999))
print(s.recv(1024).decode('utf-8'))
for data in [b'Michael', b'Tracy', b'Sarah']:
    s.send(data)
    print(s.recv(1024).decode('utf-8'))
s.send(b'exit')
s.close()

#s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')
# receive data

buffer = []
while True:
  d = s.recv(1024)
  if d:
    buffer.append(d)
  else:
    break
data = b''.join(buffer)

s.close()

header, html = data.split(b'\r\n\r\n', 1)
print(header.decode('utf-8'))
with open('sina.html','wb') as f:
  f.write(html)
  
#服务器

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind('127.0.0.1', 9999)
s.listen(5)
print('Waiting for connection...')

while True:
    socket, addr = s.accept()
    t = threading.Thread(target= tcplink, args=(socket,addr))
    t.start()
    
def tcplink(sock,addr):
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome')
    while True:
          data = sock.recv(1024)
          time.sleep(1)
          if not data or data.decode('utf-8') == 'exit':
              break
          sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)
    

