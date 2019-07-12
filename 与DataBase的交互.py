#主要讨论关系型数据库 和关系型数据库管理系统(RDBMS)
#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sqlite3

#db01_count_rows.py
#通用的CRUD  create、read、update、delete

#创建带有4个属性的sales表
#创建一个代表数据库的连接对象, 临时的
# con = sqlite3.connect(':memory:')
# #如果想要可持久化的数据库
# # con = sqlite3.connect('my_database.db) || sqlite.connect('C:\Users\linln\Desktop\my_database.db')
#
# #该字符串是一个SQL命令
# query = '''CREATE TABLE sales
#             (customer VARCHAR(20),
#             product VARCHAR(40),
#             amount FLOAT,
#             date DATE);'''
#
# con.execute(query)
# con.commit()
#
#
# # 在表中插入几行数据
# data = [('Richard Lucas', 'Notepad', 2.50, '2019-07-12'),
#         ('Jenny Kim', 'Binder', 4.15, '2019-07-12'),
#         ('Svetlana Crow', 'Printer', 155.75, '2018-02-03'),
#         ('Stephen Randolph', 'Computer', 670.40, '2018-2-20')]
# statement = 'INSERT INTO sales VALUES(?,?,?,?)'
# con.executemany(statement, data)
# con.commit()
#
# #查询sales表
# cursor = con.execute('SELECT * FROM sales')
# rows = cursor.fetchall()
#
# #计算查询结果中行的数量
# row_counter = 0
# for row in rows:
#     print(row)
#     row_counter += 1
# print('Number of rows: %d' % (row_counter))

#db02_insert_row.py
import csv, sys

# input_file = sys.argv[1]
# con = sqlite3.connect(':Suppliers:')
# c = con.cursor()
# create_table = '''CREATE TABLE IF NOT EXITS Suppliers
#                   (Supplier_Name VARCHAR(20)
#                    Invoice_Number VARCHAR(20),
#                    Part_Number VARCHAR(20),
#                    Cost FLOAT,
#                    Purchase_Date DATE'''
#
# c.execute(create_table)
# c.commit()
#
# file_reader = csv.reader(open(input_file, 'r'), delimiter =',')
# header = next(file_reader, None)
# for row in file_reader:
#     data =[]
#     for col_index in range(len(header)):
#         data.append(row[col_index])
#     print(data)
#     c.execute('INSERT INTO Suppliers VALUES(?,?,?,?,?);', data)
# con.commit()
# print('')
#
# output = c.execute('SELECT * FROM Suppliers')
# rows = output.fetchall()
# for row in rows:
#     output = []
#     for col_index in range(len(row)):
#         output.append(str(row[col_index]))
#     print(output)
#

#db03_update_row.py
# input_file = sys.argv[1]
#
# con = sqlite3.connect(':memory:')
# query = '''CREATE TABLE IF NOT EXITS sales
#             (customer VARCHAR(20)
#              product VARCHAR(40)
#              amount FLOAT
#              date DATE'''
# con.execute(query)
# con.commit()
#
# data = [('Richard Lucas', 'Notepad', 2.50, '2019-07-12'),
#         ('Jenny Kim', 'Binder', 4.15, '2019-07-12'),
#         ('Svetlana Crow', 'Printer', 155.75, '2018-02-03'),
#         ('Stephen Randolph', 'Computer', 670.40, '2018-2-20')]
# for tuple in data:
#     print(tuple)
# statement = 'INSERT INTO sales VALUES(?,?,?,?)'
# con.executemany(statement, data)
# con.commit()
#
# file_reader = csv.reader(open(input_file,'r'),delimiter = ',')
# header = next(file_reader,None)
# for row in file_reader:
#     data = []
#     for col_index in range(len(header)):
#         data.append(row[col_index])
#     print(data)
#     con.execute('UPDATE sales SET amount=?, date=? WHERE customer=?;', data)
# con.commit()
#
# cursor = con.execute('SELECT * FROM sales')
# rows = cursor.fetchall()
# for row in rows:
#     output = []
#     for col_index in range(len(row)):
#         output.append(str(row[col_index]))
#     print(output)

#db04_mysql_load_from_csv.py
# import MySQLdb
# import csv
# from datetime import datetime, date
# 
# input_file = sys.argv[1]
# 
# con = MySQLdb .connect(host='loaclhost', port=3306, db='my_suppliers', user='root', passwd='my_password')
# c = con.cursor()
# 
# filereader = csv.reader(open(input_file,'r'),delimiter =',')
# header = next(filereader, None)
# for row in filereader:
#     data = []
#     for col_index in range(len(row)):
#         if col_index < 4:
#             data.append(str(row[col_index]).lstrip('$').replace(',','').strip())
#         else:
#             a_date = datetime.date(datetime.strptime(str[row[col_index], '%m/%d/%Y']))
#             a_date = a_date.strftime('%Y-%m-%d')
#             data.append(a_date)
#     print(date)
#     c.execute('''INSERT INTO Suppliers VALUE (%s,%s,%s,%s,%s);''',data)
# con.commit()
# print('')
# c.execute('SELECT * FROM Suppliers')
# rows = c.fetchall()
# for row in rows:
#     row_list_output = []
#     for col_index in range(len(row)):
#         row_list_output.append(str(row[col_index]))
#     print(row_list_output)

#查询一个表并将输出写入CSV文件
import csv
import MySQLdb
import sys

output_file = sys.argv[1]
con = MySQLdb.connect(host='localhost', port=3306, db='my_suppliers', user='root', password='my_password')
c = con.cursor()

filewriter = csv.writer(open(output_file,'w',newline=''),delimiter=',')
header = ['Supplier Name', 'Invoice Number', 'Part Number','Cost','Purchase Date']
filewriter.writerow(header)

c.execute('''SELECT * FROM Suppliers WHERE Cost> 700.0;''')
rows = c.fetchall()
for row in rows:
    filewriter.writerow(row)
    

#更新表中的记录
input_file = sys.argv[1]
con = MySQLdb.connect(host='localhost',port = 3306, db='my_suppliers', user='root', passwd='my_password')
c = con.cursor()

file_reader = csv.reader(open(input_file,'r',newline=''), delimiter = ',')
header = next(file_reader, None)
for row in file_reader:
    data = []
    for col_index in range(len(header)):
        data.append(str(row[col_index]).strip())
    print(data)
    c.execute('''UPDATE Suppliers SET Cost=%s, Purchase_Date=%s WHERE Supplier_Name = %s;''',data)
con.commit()

c.execute('SELECT * FORM Suppliers')
rows = c.fetchall()
for row in rows:
    output = []
    for col_index in range(len(row)):
        output.append(str(row[col_index]))
    print(output)

