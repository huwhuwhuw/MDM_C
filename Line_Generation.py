#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:30:04 2025

@author: huwtebbutt
"""
import numpy as np
import math
from matplotlib import pyplot as plt

def Move_Forward(Point,Dict,Step=0.005,Attractor_Strength=4,Collect_Distance=0.001):
    def Get_Angle(point1, point2):#Gets Angle from 1 point to another
        delta_x=point2[0]-point1[0]
        delta_y=point2[1]-point1[1]
        Angle=math.atan2(delta_y,delta_x)
        return Angle
    def Average_Point(Point,Dict):
        #Calculates the Average location of all the points in the dictionary
        #taking into account the distance based weights,
        #Removes points that the line has already hit
        Average_x=[]
        Average_y=[]
        KILL=[]
        n=0
        #For every point in the dictionary
        for Key,Vals in zip(Dict.keys(),Dict.values()):
            Coords=Vals[1]
            #Distance between Line point and Dict point
            Distance=((Point[0]-Coords[0])**2 + (Point[1]-Coords[1])**2)**0.5
            if Distance <= Collect_Distance: #List of points within kill radius
                KILL.append(Key)
            #Weight = Arrival Departure Weight, + how close the point is
            Weight=sum(Vals[0]) / Distance**Attractor_Strength
            
            #Shift coords for no reason
            Shifted_x=Coords[0]-Point[0]
            Shifted_y=Coords[1]-Point[1]
            Average_x.append(Shifted_x*Weight)
            Average_y.append(Shifted_y*Weight)
            n+=Weight
        Average_x=sum(Average_x)/n
        Average_y=sum(Average_y)/n
        #Remove points that tram will have gone near to stop local minima
        for Values_To_Kill in KILL:
            Dict.pop(Values_To_Kill)
        return Average_x,Average_y
            
    x,y=Average_Point(Point,Dict)
    #Find angle from current point to weighted average point
    Angle=Get_Angle((0,0),(x,y))
    #New point = current point + step in direction of weighted average point
    New_Point= (Point[0]+Step*np.cos(Angle), Point[1]+Step*np.sin(Angle))
    return New_Point



Center=(51.4545, -2.5879)

filepath = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Bris_Codes_with_Weights_and_Coords_NEW.json'
with open(filepath, 'r') as file:
    import json
    Dict = json.load(file)

Vals=Dict.values()
Lat=[]
Long=[]
Size=[]
for val in Vals:
    Lat.append(val[1][0])
    Long.append(val[1][1])
    Size.append(sum(val[0])/400)
#Get rough radius
Radius=max(sum(Lat)/len(Lat)-min(Lat),sum(Long)/len(Long)-min(Long))

fig,ax=plt.subplots(dpi=500)
ax.scatter(Long,Lat,s=Size,c='k')

#Number of lines
n=10
for i in range(n):
    #Create copy of dict so killing values doesnt fuck everything
    Copy_Dict=Dict.copy()
    
    #Get angles to place starting station
    i_angle=i*2*np.pi /n
    Start_Lat=Center[0]+Radius*np.cos(i_angle)
    Start_Long=Center[1]+Radius*np.sin(i_angle)
    Start_Point=(Start_Lat,Start_Long)
    
    Points=[Start_Point]
    #Distance between current point and end goal (Center)
    D=((Start_Point[0]-Center[0])**2+(Start_Point[1]-Center[1])**2)**0.5
    
    count=1
    while  count<10000 and D>0.008: # Iterate until reaches center, failsafe of 10000 steps
        New_point=Move_Forward(Points[-1],Copy_Dict,Step=0.001,Attractor_Strength=4,Collect_Distance=0.006)
        Points.append(New_point)
        D=((New_point[0]-Center[0])**2+(New_point[1]-Center[1])**2)**0.5
        count+=1
    
    #Plot line
    Points=np.array(Points).T
    ax.plot(Points[1],Points[0],lw=1)
    
    