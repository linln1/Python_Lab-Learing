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
                  
            
