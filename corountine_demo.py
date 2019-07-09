# usr/bin/env python3
# -*- encoding:utf-8 -*-

import asyncio

@asyncio.coroutine
def hello():
      print('Hello, world! (%s)' % threading.currentThread())
      yield from asyncio.sleep(1)
      print('Hello, again! (%s)' % threading.currentThread())
      

loop = asyncio.get_loop_event()
task = [hello(), hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()


def getwebpage(host):
      print('getwebpage %s...' % host)
      connect = asyncio.open_connection(host, 80)
      reader, writer = yield from connect
      header  = 'GET / HTTP/1.0\r\nHost:  %s\r\n\r\n' % host
      writer.write(header.encode('utf-8')
      yield from writer.drain()
      while True:
            line = yield from reader.readline()
            if line == b'\r\n':
                  break
            print('%s header > %s' % (host, line.decode('utf-8').rstrip())
      write.close()
      
loop = asyncio.get_loop_event()
tasks = [getwebpage(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()

以下为等价写法
# async def hello():
#                   print('hello,world! (%s)' % threading.currentThread())
#                   r = await asyncio.sleep(1)
#                   print('hello,again! (%s)' % threading.currentThread())
                  
