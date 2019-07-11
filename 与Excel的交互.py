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

#格式化日期
from datetime import date
from xlrd import open_workbook, xldate_as_tuple
form xlwt import Workbook
input
