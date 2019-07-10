#! /usr/bin/env python3
#-*- encoding utf-8 -*-

#example 1
import os, time, random, sys

my_numbers = [0,1,2,3,4,5,6,7,8,9]
max_index = len(my_numbers)
output_file = sys.argv[1]
filewriter = open(output_file,'a')
for index in range(max_index):
      if index < (max_index - 1):
            filewriter.write(str(my_numbers[index]) + ',')
      else:
            filewriter.write(str(my_numbers[index]) + '\n')
filewriter.close()


#example 2 
import os, time, random, sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file,'r', newline='') as filereader:
      with open(out_file, 'w', newline='') as filewriter:
            header = filereader.readline()
            header = header.strip()
            header_list = header.split(',')
            print(header_list)
            filewriter.write(','.join(map(str,header_list))+'\n')
            for row in filereader:
                  row = row.strip()
                  row_list = row.split(',')
                  print(row_list)
                  filewriter.write(','.join(map(str,row_list))+'\n')
                  
#！ /usr/bin/env python
# -*- encoding:utf-8 -*-

import csv
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

# with open(input_file,'r',newline='') as csv_in_file:
#     with open(output_file, 'w',newline='')as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         header = next(filereader)
#         filewriter.write(header)
#         for row_list in filereader:
#             supplier = str(row_list[0]).strip()
#             cost = str(row_list[3]).strip('$').replace(',','')
#             if supplier == 'Supplier Z' or float(cost) > 600.0:
#                 filewriter.writerow(row_list)



#pandas

import pandas as pd

# important_dates = ['1/20/14', '1/30/14']
#
# with open(input_file, 'r', newline='') as csv_in_file:
#     with open(output_file, 'w', newline=) as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         header = next(filereader)
#         filewriter.write(header)
#         for row_list in filereader:
#             print(row_list)
#             a_date = row_list[4]
#             if a_data in important_dates:
#                 filewriter.writerow(a_date)
#
# data_frame = pd.read_csv(input_file)
# data_frame_value_in_set = data_frame.loc[data_frame['Purchase Date'].isin(important_dates), :]
# data_frame_value_in_set.to_csv(output_file, index = False)

# pattern = re.compile(r'(?P<my_pattern_group>^001-.*)', re.I)
# with open(input_file,'r', newline='') as csv_in_file:
#     with open(output_file,'w',newline='') as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         heaeder = next(filereader)
#         filewriter.write(header)
#         for row_list in filereader:
#             invoice_num  = row_list[1]
#             if pattern.search(invoice_num):
#                 filewriter.writerow(row_list)
#
# #pandas
# data_frame = pd.read_csv(input_file)
# data_frame_value_matches_pattern = data_frame.loc[data_frame['Invoice_num'].str.stratwith('001-'), :]
# data_frame_value_matches_pattern.to_csv(output_file, index = False)
#
# #选取特定的列
#
# #csv06_reader_column_by_index.py
# # 列索引
# my_columns = [0,3]
# with open(input_file,'r',newline='') as csv_in_file:
#     with open(output_file,'w',newline='') as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         header = next(filereader)
#         filewriter.write(header)
#         for row_list in filereader:
#             row_list_output = []
#             for index_value in my_columns:
#                 row_list_output.append(row_list[index_value])
#             filewriter.writerow(row_list_output)
#
#
# data_frame = pd.read_csv(input_file)
# data_frame_column_by_index = data_frame.iloc[:, my_columns]
# data_frame_column_by_index.to_csv(output_file, index=False)
#
# #csv07_reader_column_by_name.py
# sec_columns = ['Invoice number', 'Purchase Date']
# sec_columns_index = []
# with open(input_file,'r',newline='') as csv_in_file:
#     with open(output_file,'w',newline='') as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         header = next(filereader, None)
#         for index_value in range(len(header)):
#             if header[index_value] in sec_columns:
#                 sec_columns_index.append(index_value)
#         filewriter.writerow(sec_columns)
#         for row_list in filereader:
#             row_list_output = []
#             for index_value in sec_columns_index:
#                 row_list_output.append(row_list[index_value])
#             filewriter.writerow(row_list_output)
#
# #pandas
# data_frame = pd.read_csv(input_file)
# data_frame_column_by_name = data_frame.loc[:, ['Invoice number', 'Purchase Date']]
# data_frame_column_by_name.to_csv(output_file, index = False)
#
# #选取连续的行
# row_counter >= 3 and row_counter <=15
# filewriter.writerow([value.strip() for value in row])
# row_counter += 1
#
#
# #pandas
# data_frame = pd.read_csv(input_file, header = None)
# data_frame = data_frame.drop([0,1,2,16,17,18])
# data_frame.columns = data_frame.iloc[0]
# data_frame = data_frame.reindex(data_frame.index.drop(3))
# data_frame.to_csv(output_file, index = False)
#
# #添加行标题
# with open(input_file,'r',newline='') as csv_in_file:
#     with open(output_file,'w',newline='') as csv_out_file:
#         filereader = csv.reader(csv_in_file)
#         filewriter = csv.writer(csv_out_file)
#         header_list = ['Supplier name', 'Invoice number', 'Part number', 'Cost', 'Purchase Date']
#         filewriter.writerow(header_list)
#         for row in filereader:
#             filewriter.writerow(row)
#
# #pandas
#
# header_list = ['Supplier name', 'Invoice number', 'Part number', 'Cost', 'Purchase Date']
# data_frame = pd.read_csv(input_file, header = None, names=header_list)
# data_frame.to_csv(output_file, index = False)

#读取多个CSV文件

import csv, sys, os, glob
input_path = sys.argv[1]
file_counter = 0

for input_file in glob.glob(os.path.join(input_path, 'sales_*')):
    row_countor = 1
    with open(input_file, 'r', newline='')as csv_in_file:
        filereader = csv.reader(csv_in_file)
        header = next(filereader, None)
        for row in filereader:
            row_countor += 1
            print('{0!s}: \t{1:d} row \t{2:d} columns'.format(os.path.basename(input_file), row_countor, len(header)))
    file_counter +=1
print('Numbers of files : {0:d}'.format(file_counter))

#从多个文件中链接数据

#csv09_reader_concat_rows_from_multiple_files.py

input_path = sys.argv[1]
output_path = sys.argv[2]

first_file = True
for input_file in glob.glob(os.path.join(input_path,'sales_*')):
    print(os.path.basename(input_file))
    with open(input_file, 'r', newline='')as csv_in_file:
        with open(output_file, 'a',newline='') as csv_out_file:
            filereader = csv.reader(csv_in_file)
            filewriter = csv.writer(csv_out_file)
            if first_file:
                 for row in filereader:
                     filewriter.writerow(row)
                 first_file = False
            else:
                 heaeder = next(filereader, None)
                 for row in filereader:
                     filewriter.writerow(row)


#pandas
all_files = glob.glob(os.path.join(input_path, 'sales_*'))
all_data_frams = []
for file in all_files:
    data_fram = pd.read_csv(file, index_col= None)
    all_data_frams.append(data_fram)
data_fram_concat = pd.concat(all_data_frams, axis=0, ignore_index = True)
data_fram_concat.to_csv(output_file, index=False)


#计算每个文件中值的总和和均值
#csv10_reader_sum_average_from_multiple_files:

input_path = sys.argv[1]
output_path = sys.argv[2]
output_header_list = ['file_name', 'total_sales', 'average_sales']
csv_out_file = open(output_file, 'a', newline='')
filewriter = csv.writer(csv_out_file)
filewriter.writerow(output_header_list)
for input_file in glob.glob(os.path.join(input_path, 'sales_*')):
    with open(input_file, 'r', newline='') as csv_in_file:
        filereader = csv.reader(csv_in_file)
        output_list = []
        output_list.append(os.path.basename(input_file))
        header = next(filereader)
        total_sales = 0.0
        numbers_of_sales = 0.0
        for row in filereader:
            sale_amount = row[3]
            total_sales += float(str(sale_amount).strip('$').replace(',',''))
            numbers_of_sales += 1
        average_sales ='{0:.2f}'.format(total_sales/numbers_of_sales)
        output_list.append(total_sales)
        output_list.append(average_sales)
        filewriter.writerow(output_list)
csv_out_file.close()

#pandas
all_file = glob.glob(os.path.join(input_path,'sales_*'))
all_data_frams = []
for input_file in all_file:
    data_fram = pd.read_csv(input_file, index_col = None)
    total_cost = pd.DataFrame([float(str(value).strip('$').replace(',','')) for value in data_fram.loc[:, 'Sales Amount']]).sum()
    average_cost = pd.DataFrame([float(str(value).strip('$').replace(',','')) for value in data_fram.loc[:, 'Sale Amount']]).mean()
    data = {'file_name':os.path.basename(input_file),
            'total_sales': total_sales,
            'average_sales': average_sales}
    all_data_frams.append(pd.DataFrame(data, columns=['file_name', 'total_sales','average_sales']))
    data_fram_concat = pd.concat(all_data_frams, axis=0, ignore_index= True)
    data_fram_concat.to_csv(output_file, index=False)
    
                                 
        
            
