#! /usr/bin/env python
# -*- encoding:utf-8 -*-

import sys, os
from xlrd import open_workbook
input_file = sys.argv[1]
workbook = open_workbook(input_file)
print('Numbers of worksheets:', workbook.nsheets)
for worksheet in workbook.sheets():
      print('Worksheet name:', worksheet.name, '\tRows:', worksheet.nrows, '\tColumns:', worksheet.ncols)
      
#处理单个工作表
from xlrd import open_workbook
from xlwt import Workbook
input_file = sys.argv[1]
output_file = sys.argv[2]
output_workbook = Workbook()
output_worksheet = output_workbook.add_sheet('jul_2019_output')
with open_workbook(inout_file) as workbook:
      worksheet = workbook.sheet_by_name('jul_2019')
      for row_index in range(worksheet.nrows):
            for col_index in range(worksheet.ncols):
                  output_worksheet.write(row_index, col_index, worksheet.cell_value(row_index, col_index))
output_workbook.save(output_file)

# -*- coding:utf-8 -*-

from datetime import date
from xlrd import open_workbook, xldate_as_tuple
from xlwt import Workbook
import pandas as pd


input_file = sys.argv[1]
output_file = sys.argv[2]
output_workbook = Workbook()
output_worksheet = output_workbook.add_sheet('Jul_2019_output')
with open_workbook(input_file) as workbook:
    worksheet = workbook.sheet_by_name('Jul_2019')
    for row_index in range(worksheet.nrows):
        row_list_output = []
        for col_index in range(worksheet.ncols):
            if worksheet.cell_type(row_index,col_index)==3:
                date_cell = xldate_as_tuple(worksheet.cell_value(row_index, col_index), workbook.datemode)
                date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                row_list_output.append(date_cell)
                output_worksheet.write(row_index, col_index, date_cell)
            else:
                non_date_cell = worksheet.cell_value(row_index, col_index)
                row_list_output.append(non_date_cell)
                output_worksheet.write(row_index, col_index, non_date_cell)


#pandas
data_frame = pd.read_excel(input_file, sheetname='july_2019')
writer = pd.ExcelWriter(output_file)
data_frame.to_excel(writer, sheet_name='July_2019_output', index=False)
writer.save()

#筛选特定行
sale_amount_column_index = 3
with open_workbook(input_file)as workbook:
    worksheet = workbook.sheet_by_name('July_2019_output')
    data = []
    header = worksheet.row_values(0)
    data.append(heaeder)
    for row_index in range(1,worksheet.nrows):
        row_list = []
        sale_amount = worksheet.cell_value(row_index, sale_amount_column_index)
        if sale_amount > 1400.0:
            for col_index in range(1,worksheet.ncols):
                cell_value = worksheet.cell_value(row_index, col_index)
                cell_type = worksheet.cell_type(row_index, col_index)
                if cell_type == 3:
                    date_cell = xldate_as_tuple(cell_value,workbook.datemode)
                    date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                    row_list.append(date_cell)
                else:
                    row_list.append(cell_value)
        if row_list:
                data.append(row_list)
    for list_index, output_list in enumerate(data):
        for element_index,element in enumerate(output_list):
                output_worksheet.write(list_index,element_index,element)
output_worksheet.save(output_file)

data_frame = pd.read_excel(input_file, 'July_2019', index_col= None)
data_frame_value_meets_condition = data_frame[data_frame['Sale Amount'].astype(float)> 1400.0]
writer = pd.ExcelWriter(output_file)
data_frame_value_meets_condition.to_excel(writer, sheet_name = 'July_2019_output', index=False)
writer.save()

#行中的值属于某个集合

output_workbook = Workbook()
output_worksheet = output_workbook.add_sheet('July_2019_output')
important_dates = ['01/24/2019','01/31/2019']
purchase_date_column_index = 4
with open_workbook(input_file) as workbook:
    worksheet = workbook.sheet_by_name('July_2019')
    data = []
    header = worksheet.row_values(0)
    data.append(header)
    for row_index in range(1, worksheet.nrows):
        purchase_datetime = xldate_as_tuple(worksheet.cell_value(row_index, purchase_date_column_index, workbook.datemode))
        row_list = []
        if purchase_date in important_dates:
        for col_index in range(1, worksheet.ncols):
            cell_value = worksheet.cell_value(row_index, col_index)
            cell_type = worksheet.cell_type(row_index, col_index)
            if cell_type == 3:
                date_cell = xldate_as_tuple(cell_value, workbook.datemode)
                date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                row_list.append(date_cell)
            else:
                row_list.append(cell_value)
    if row_list:
        data.appent(row_list)
    for list_index, output_list in enumerate(data):
        for element_index, element in enumerate(output_list):
            output_worksheet.write(list_index, element_index, element)
output_worksheet.save(output_file)

#pandas
data_frame = pd.read_excel(input_file, 'July_2019', index_col = None)
data_frame_value_in_set = data_frame[data_frame['PurchaseDate'].isin(important_dates)]
writer = pd.ExcelWriter(output_file)
data_frame_value_in_set.to_excel(writer, sheet_name = 'July_2019_output', index= False)
writer.save()

#行中的值匹配特定模式
import re

pattern = re.compile(r'(?P<my_pattern>^J.*)')
customer_name_column_index = 1
with open_workbook(input_file) as workbook:
    worksheet = workbook.sheet_by_name('July_2019')
    data = []
    header = worksheet.row_values(0)
    data.append(header)
    for row_index in range(1, worksheet.nrows):
        row_list = []
        if pattern.search(worksheet.cell_value(row_index, customer_name_column_index)):
            for col_index in range(worksheet.ncols):
                cell_value = worksheet.cell_value(row_index, col_index)
                cell_type = worksheet.cell_type(row_index,col_index)
                if cell_type == 3:
                    date_cell = xldate_as_tuple(cell_value, workbook.datemode)
                    date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                    row_list.append(date_cell)
                elseL
                row_list.append(cell_value)
    if row_list:
        data.append(row_list)
    for list_index, output_list in enumerate(data):
        for element_index, element in enumerate(output_list):
            output_worksheet.write(list_index, element_index, element)
output_worksheet.save(output_file)


#选取列标题
#pandas

data_frame = pd.read_excel(input_file, 'July_2019', index_col = None)
data_frame_column_by_name = data_frame.loc[:,['Customer ID', 'Purchase Date']]
writer = pd.ExcelWriter(output_file)
data_frame_column_by_name.to_excel(writer, sheet_name = 'July_2019_output', index = False)
writer.save()

#读取工作簿中的所有工作表
#pandas

data_frame = pd.read_excel(input_file, sheetname=None, index_col = None)
row_output = []
for worksheet_name, data in data_frame.items():
    row_output.append(data[data['Sale Amount'].astype(float)>2000.0])
filtered_rows = pd.concat(row_output, axis = 0, ignore_index=True)
writer = pd.ExcelWriter(output_file)
filtered_rows.to_excel(writer, sheet_name = 'Sale Amount_gt2000', index=False)
writer.save()

#所有工作表中选取特定列
import sys
from xlrd import  open_workbook, xldate_as_tuple
from xlwt import Workbook
import pandas as pd

input_file = sys.argv[1]
output_file = sys.argv[2]
workbook = Workbook()
worksheet = workbook.add_sheet('July_2019')
my_columns = [1,4]
with open_workbook(input_file) as workbook:
    worksheet = workbook.sheet_by_name('July_2019')
    data = []
    for row_index in range(worksheet.nrows):
        row_list = []
        for col_index in my_columns:
            cell_value = worksheet.cell_value(row_index, col_index)
            cell_type = worksheet.cell_type(row_index, col_index)
            if cell_type == 3:
                data_cell = xldate_as_tuple(cell_value, workbook.datemode)
                date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                row_list.append(date_cell)
            else:
                row_list.append(cell_value)
        data.append(row_list)
    for list_index, output_list in enumerate(data):
        for element_index, element in enumerate(output_list):
            output_worksheet.write(list_index, element_index, element)
output_workbook.save(output_file)

#pandas
input_file = sys.argv[1]
output_file = sys.argv[2]
data_frame = pd.read_excel(input_file, 'July_2019', index_col=None)
data_frame_column_by_index = data_frame.iloc[:,[1,4]]
writer = pd.ExcelWriter(output_file)
data_frame_column_by_index.to_excel(writer, 'July_2019_output_file', index = False)
writer.save()

#列标题
input_file = sys.argv[1]
output_file = sys.argv[2]
output_workbook = Workbook()
output_worksheet = output_workbook.add_sheet('July_2019_output')
my_columns = ['Customer ID', 'Purchase Date']
with open_workbook(input_file) as workbook:
    worksheet = workbook.sheet_by_name('July_2019')
    data = [my_columns]
    header_list = worksheet.row_values(0)
    header_index_list = []
    for header_index in range(len(header_list)):
        if header_list[header_index] in my_columns:
            header_index_list.append(header_index)
    for row_index in range(1,worksheet.nrows):
        row_list = []
        for col_index in header_index_list:
            cell_value = worksheet.cell_value(row_index, col_index)
            cell_type = worksheet.cell_type(row_index, col_index)
            if cell_type == 3:
                data_cell = xldate_as_tuple(cell_value, workbook.datemode)
                date_cell = date(*date_cell[0:3]).strftime('%m/%d/%Y')
                row_list.append(date_cell)
            else:
                row_list.append(cell_value)
        data.append(row_list)
    for list_index, output_list in enumerate(data):
        for element_index, element in enumerate(output_list):
            output_worksheet.write(list_index, element_index, element)
output_worksheet.save(output_file)

#pandas
input_file = sys.argv[1]
output_file = sys.argv[2]

data_frame = pd.read_excel(input_file, 'July_2019', index_col= None)
data_frame_column_by_name = data_frame.loc[:,['Customer ID', 'Purchase Date']]
writer = pd.ExcelWriter(output_file)
data_frame_column_by_name.to_excel(writer, sheet_name = 'July_13_output',index = False)
writer.save()

#读取工作簿中的所有工作表
#pandas
import pandas as pd
data_frame = pd.read_excel(input_file, sheetname=None, index_col=None)
row_output = []
for worksheet_name,data in data_frame.items():
    row_output.append(data[data['Sale Amount'].astype(float) > 2000.0])
filtered_rows = pd.concat(row_output, axis=0, ignore_index = True)
writer = pd.ExcelWriter(output_file)
filtered_rows.to_excel(writer, sheet_name='sale_amount_gt2000', index = False)
writer.save()

#在所有工作表中选取特定列
from datetime import date
from xlrd import open_workbook, xldate_as_tuple
from xlwt import Workbook
input_file = sys.argv[1]
output_file = sys.argv[2]
output_workbook = Workbook()
output_worksheet = output_workbook.add_sheet('set_of_worksheets')
my_sheets = [0,1]
threshold = 1900.0
sales_column_index = 3
first_worksheet = True
with open_workbook(input_file) as workbook:
    data = []
    for sheet_index in range(workbook.nsheets):
        if sheet_index in my_sheets:
            worksheet = workbook.sheet_by_name(sheet_index)
            if first_worksheet:
                header_row = workbook.row_values(0)
                data.append(header_row)
                first_worksheet = False
            for row_index in range(workbook.nrows):
                row_list = []
                sales_amount = worksheet.cell_value(row_index, sales_column_index)
                if sales_amount > threshold:
                    for col_index in range(worksheet.ncols):
                        cell_value = worksheet.cell_value(row_index, col_index)
                        cell_type = worksheet.cell_type(row_index, col_index)
                        if cell_type == 3:
                            date_cell = xldate_as_tuple(cell_value, workbook.datemode)
                            date_cell = date[*date_cell[0:3]].strftime('%m/%d/%Y')
                            row_list.append(date_cell)
                        else:
                            row_list.append(cell_value)
                if row_list:
                    data.append(row_list)

    for list_index, output_list in enumerate(data):
        for element_index, element in enumerate(output_list):
            output_worksheet.write(list_index, element_index, element)
output_worksheet.save(output_file)


#处理多个工作簿
#工作表计数以及每个工作表中的行列计数

import glob,os, sys
input_directory = sys.argv[1]
workbook_counter = 0
for input_file in glob.glob(os.path.join(input_directory, '*,xls*')):
    workbook = open_workbook(input_file)
    print('Workbook: %s' % os.path.basename(input_file))
    print('NUmber of worksheets: %d' % workbook.nsheets)
    for worksheet in workbook.sheets():
        print('Worksheet name:', worksheet.name, '\tRows:', worksheet.nrows, '\tColumns:', workbook.ncols)

    workbook_counter += 1
print('Number of Excel workboos: %d' % (workbook_counter))

#从多个工作簿中连接数据


