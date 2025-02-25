#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:32:01 2025

@author: huwtebbutt
"""

import csv

# Replace 'your_file.csv' with the path to your CSV file
#file_path = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Output_area_(2021)_to_future_Parliamentary_Constituencies_Lookup_in_England_and_Wales.csv'
file_path = ''


Bristol_Codes=[]

with open(file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if 'Bristol, City of' in row:
            Bristol_Codes.append(row[0])

for code in Bristol_Codes:
    print(code)

#file_path= '/Users/huwtebbutt/Documents/MDM3/3rd_term/odwp01ew/ODWP01EW_OA.csv'
file_path= ''


Bristol_Rows=[]
with open(file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    new_row=[]
    for row in csv_reader:
        if row[0] in Bristol_Codes and row[2] in Bristol_Codes:
            new_row=['B',row[0],'B',row[2],row[-1]]
            Bristol_Rows.append(new_row)
        elif row[0] in Bristol_Codes:
            new_row=['B',row[0],'NB',row[2],row[-1]]
            Bristol_Rows.append(new_row)
        elif row[2] in Bristol_Codes:
            new_row=['NB',row[0],'B',row[2],row[-1]]
            Bristol_Rows.append(new_row)
        

#filename = '/Users/huwtebbutt/Documents/MDM3/3rd_term/odwp01ew/Bristol_OA_Data.csv'
filename = ''

# Writing to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(Bristol_Rows)