#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:02:45 2025

@author: huwtebbutt
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

def CSV_to_Dict(CSV_path):
    # Make Dictionary where OA code is the key, and coords are the value
    Coords={}
    with open(CSV_path, mode='r', newline='') as file:
        print()
        csv_reader = csv.reader(file)
        for row in csv_reader:
            Coords[row[0]]=(row[1],row[2])
    return Coords


Coords=CSV_to_Dict('/Users/huwtebbutt/Documents/MDM3/3rd_term/Bristol_Codes_With_Coords.csv')

# Sample data
OA_data='/Users/huwtebbutt/Documents/MDM3/3rd_term/odwp01ew/Bristol_OA_Data.csv'

# Get Departures, arrivals and weights
with open(OA_data,mode='r',newline='') as file:
    departures=[]
    arrivals=[]
    weights=[]
    csv_reader = csv.reader(file)
    for row in csv_reader:
        departures.append(row[0])
        arrivals.append(row[1])
        weights.append(row[2])
        
# Calculate frequency of each point with weights
weighted_counts_d = {}
weighted_counts_a = {}
for point_1,point_2, weight in zip(departures,arrivals, weights):
    if point_1 in weighted_counts_d:
        weighted_counts_d[point_1] += float(weight)
    else:
        weighted_counts_d[point_1] = float(weight)
    
    if point_2 in weighted_counts_a:
        weighted_counts_a[point_2] += float(weight)
    else:
        weighted_counts_a[point_2] = float(weight)

unique_d = np.array(list(weighted_counts_d.keys()))
counts_d = np.array(list(weighted_counts_d.values()))
sizes_d = [float(count)/400 for count in counts_d]  # Scale marker sizes

unique_a = np.array(list(weighted_counts_a.keys()))
counts_a = np.array(list(weighted_counts_a.values()))
sizes_a = [float(count)/400 for count in counts_a]  # Scale marker sizes

x_1=[]
y_1=[]
x_2=[]
y_2=[]

# Look up coords for each OA code
for code_d,code_a in zip(unique_d,unique_a):
    lat,long=Coords[code_d]
    y_1.append(float(lat))
    x_1.append(float(long))
    
    lat,long=Coords[code_a]
    y_2.append(float(lat))
    x_2.append(float(long))

# Scatter plot
fig_1,ax_1=plt.subplots(dpi=500)
fig_2,ax_2=plt.subplots(dpi=500)

ax_1.scatter(x_1,y_1,s=sizes_d,alpha=1)
ax_2.scatter(x_2,y_2,s=sizes_a,alpha=1)

ax_1.set_title('Departures')
ax_2.set_title('Arrivals')

ax_1.set_axis_off()
ax_2.set_axis_off()

# Attempt at clustering
from Clustering_Algorithms import DBScan_Cluster

points_1=np.array((x_1,y_1)).T
points_2=np.array([x_2,y_2]).T


DBScan_Cluster(points_1,eps=10,min_samples=1000)
DBScan_Cluster(points_2,)



