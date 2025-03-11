#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Takes 2011_OAs_Bath_North_Somers_Cleaned.csv, or 2011_OAs_Bris_GlousNorth_Cleaned.csv 
To do both, run twice but second time input json file on line 86

Outputs a json file with format
Keys= OA codes
Values= [Weights, (Lat,Long)]
"""
#import pandas as pd
import requests
from geopy.distance import geodesic
import json
import csv
import time
ST=time.time()
print(ST)

def Get_Coords_From_Code(Code,Dict):
    #Check if Code already in Dict
    if Code in Dict.keys():
        Weight,Coords=Dict[Code]
        Lat,Long=Coords
        return Lat,Long
    
    #If not, get coords using API
    url = f'https://findthatpostcode.uk/areas/{Code}.json'
    #Have to get coords of area by averaging coords of example postcodes
    Example_Postcodes=[]
    response = requests.get(url)
    if response.status_code == 200:
        print(f'{Code}')
        #Navigate to Postcodes
        D = response.json()
        Data=D['data']
        Relationships=Data['relationships']
        Postcodes=Relationships['example_postcodes']
        Postcode_data=Postcodes['data']
        for post in Postcode_data:
            Ex_Post=post['id'].replace(" ", "+")
            Example_Postcodes.append(Ex_Post)
    else:
        print(f"Error: {response.status_code}\n{Code}")
    Lats=[]
    Longs=[]
    #Use API for each example postcode
    for P in Example_Postcodes:
        postcode=f'postcodes/{P}'
        url = f'https://findthatpostcode.uk/{postcode}.json'
        response = requests.get(url)
        if response.status_code == 200:
            print(P)
            #Navigate to Coords
            D = response.json()
            data=D['data']
            attr=data['attributes']
            lattitude=attr['lat']
            longitude=attr['long']
            Lats.append(lattitude)
            Longs.append(longitude)
        else:
            print(f"Error: {response.status_code}\n{P}")
    #Average postcode coords to get good guess at area coords
    Lat=sum(Lats)/len(Lats)
    Long=sum(Longs)/len(Longs)
    return Lat,Long


def is_within_radius(lat, lon, center=(51.4545, -2.5879), max_distance_km=10):
    #Returns True if within radius and False if not
    return geodesic((lat, lon), center).km <= max_distance_km



Center_Lat=51.4545
Center_Long=-2.5879

data2011='filepath/2011_OA_Bris_GlousNorth_Cleaned.csv'
#data2011='filepath/2011_OA_Bath_NorthSomers_Cleaned.csv'
file = open(data2011, mode='r', newline='')
CSV = csv.reader(file)
#df=pd.read_csv(data2011)

#Dictionary with: Key=OA Code , Values=[Weight, (Lat,Long)]. Can create new or add to existing
Filtered_Dict={}
"""filepath = 'path/path/Name.json'
with open(filepath, 'r') as file:
    Filtered_Dict = json.load(file)""" 


for i,row in enumerate(CSV):
    #Check if header row
    if i==0:
        first_row=row
        continue
    
    for j,item in enumerate(row):
        #Skip case
        if i == j:
            continue
        #if j=0 item is a code
        if j==0:
            #Get Coords and check if within 10km
            Lat,Long=Get_Coords_From_Code(item,Filtered_Dict)
            if is_within_radius(Lat,Long):
                #Add weights to dictionary
                if item in Filtered_Dict.keys():
                    Filtered_Dict[item][0]+=sum([int(weight) for weight in row[1:]])
                else:
                    Weight=sum([int(weight) for weight in row[1:]])
                    Coords=(Lat,Long)
                    Filtered_Dict[item]=[Weight,Coords]
            else:
                #Skip row if not in 10km radius
                break
        
        #j is a weight
        elif j != 0:
            Count=int(item)
            if Count==0:
                #Ignore when no commuters
                continue
            
            #Get corresponding Code and check distance
            Code=first_row[j]
            Lat,Long=Get_Coords_From_Code(Code,Filtered_Dict)
            if is_within_radius(Lat,Long):
                #Add weights to dictionary
                if Code in Filtered_Dict.keys():
                    Filtered_Dict[Code][0]+=int(item)
                else:
                    Weight=int(item)
                    Coords=(Lat,Long)
                    Filtered_Dict[Code]=[Weight,Coords]

#Save dictionary as file
filepath = 'path/path/Name.json'

#Write the dictionary to a file in JSON format
with open(filepath, 'w') as file:
    json.dump(Filtered_Dict, file)

ET=time.time()
print(ET-ST)


