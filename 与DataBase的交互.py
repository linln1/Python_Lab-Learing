#主要讨论关系型数据库 和关系型数据库管理系统(RDBMS)
#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sqlite3

#db01_count_rows.py
#通用的CRUD  create、read、update、delete

#创建带有4个属性的sales表
#创建一个代表数据库的连接对象, 临时的
con = sqlite3.connect(':memory:')
#如果想要可持久化的数据库
# con = sqlite3.connect('my_database.db) || sqlite.connect('C:\Users\linln\Desktop\my_database.db')

#该字符串是一个SQL命令
query = '''CREATE TABLE sales
            (customer VARCHAR(20),
            product VARCHAR(40),
            amount FLOAT,
            date DATE);'''

con.execute(query)
con.commit()


# 在表中插入几行数据
data = [('Richard Lucas', 'Notepad', 2.50, '2019-07-12'),
        ('Jenny Kim', 'Binder', 4.15, '2019-07-12'),
        ('Svetlana Crow', 'Printer', 155.75, '2018-02-03'),
        ('Stephen Randolph', 'Computer', 670.40, '2018-2-20')]
statement = 'INSERT INTO sales VALUES(?,?,?,?)'
con.executemany(statement, data)
con.commit()

#查询sales表
cursor = con.execute('SELECT * FROM sales')
rows = cursor.fetchall()

#计算查询结果中行的数量
row_counter = 0
for row in rows:
    print(row)
    row_counter += 1
print('Number of rows: %d' % (row_counter))

