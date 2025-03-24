#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:16:48 2025

@author: huwtebbutt
"""
import googlemaps
from geopy.distance import geodesic
import math

def Line_Score(Line,Dict):
    """
    Parameters
    ----------
    Line : List
        A list of coordinates, that correspond to the location of stations.
    Dict : Dictionary
        A dictionary with format {OA code: (Weight), (Coords)}

    Returns
    -------
    Metric : Float
        A single value that represents the efficiency of the line.
    """
    def Pounds_For_Station(Amount_of_Stations,cost_per_station=700_000):
        return Amount_of_Stations * cost_per_station
    
    def Pounds_For_Meter(Line_Length,cost_per_meter=11_500):
        return Line_Length * cost_per_meter
    
    def Road_Distance(start_coords, end_coords, api_key):
        #Get distance, make sure units are kilometers, and return float
        gmap = googlemaps.Client(key=api_key)
        route = gmap.directions(start_coords, end_coords, avoid=['highways', 'tolls'])[0]
        distance = route['legs'][0]['distance']['text']
        
        if 'km' in distance:
            distance = float(distance.replace(' km', ''))
        elif 'm' in distance:
            distance = float(distance.replace(' m', '')) /1000
        else:
            print(f'Unexpected format/n{distance}')
        
        return distance
    
    def Score(x,Walking_Distance=1):
        #Return Maximum Points if within walking distance
        #If out of walking distance, score exponentially decays
        if x<Walking_Distance:
            return 1
        else:
            return math.exp(-x+Walking_Distance)
    
    # Get Total Score of how well line services entire population
    Scores=[]
    count=0
    for Val in Dict.values():
        Coords=Val[1]
        Weight=sum(Val[0])
        Distance_to_Station=[]
        #Find min distance to nearest station
        for Station in Line:
            Station_Coords=Dict[Station][1]
            
            min_distance=geodesic(Coords,Station_Coords).km
            Distance_to_Station.append(min_distance)
        Distance_to_Station=min(Distance_to_Station)
        #Convert min distance to score
        count+=Weight
        Scores.append(Score(Distance_to_Station)*Weight)
    #Find mean score
    Avg_Score=sum(Scores)/count
    
    #Get API key from super secure location
    with open('API_KEY.txt','r') as file:    
        for row in file:
            API_key=row
    
    #Get total line length, (Assumes stations are in order in Line)
    Line_Length=0
    for i,Station in enumerate(Line):
        if i==0:
            continue
        Station_Coords=Dict[Station][1]
        Prev_Station=Line[i-1]
        Prev_Coords=Dict[Prev_Station][1]
        
        #Straight_Line=geodesic(Prev_Coords,Station_Coords).km
        Actual_Distance=Road_Distance(Prev_Coords,Station_Coords,API_key)
        Line_Length+=Actual_Distance
    
    Station_Cost= Pounds_For_Station(len(Line))
    Line_Cost  =  Pounds_For_Meter(Line_Length)
    
    Metric=Avg_Score/(Station_Cost + Line_Cost)
    return Metric