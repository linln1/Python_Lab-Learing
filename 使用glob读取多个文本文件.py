#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from math import exp, log, sqrt
import re
from datetime import data, time, datetime, timedelta
from operator import itemgetter
import sys, glob, os

input_path = sys.argv[1]

for input_file in glob.glob(os.path.join(input_path,'*.txt'))
      with open(input_file, 'r', newline='') as filereader:
            for row in filereader:
                  print('{}'.format(row.strip()))


# with open(input_file, 'r', newline='') as filereader:
#      for row in filereader:
#            print('{}'.format(row.strip()))
