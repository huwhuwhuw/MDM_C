#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 13:03:00 2025

@author: huwtebbutt
"""


def Distance_Score(Population_Dict,Station_Locations):
    from geopy.distance import geodesic
    def Get_Score(Distance):
        if Distance < 0.8:
            return 3
        elif Distance < 1.6:
            return 2
        elif Distance < 3.2:
            return 1
        else:
            return 0
    
    Total_Score=0
    List_of_Scores=[]
    for val in Population_Dict.values():
        weight=sum(val[0])
        point_coords=val[1]
        Distances=[]
        for Station_coords in Station_Locations:
            Diff=geodesic(Station_coords,point_coords).km
            Distances.append(Diff)
        Closest=min(Distances)
        Score=Get_Score(Closest)
        Total_Score+=Score*weight
        List_of_Scores.append(Score)
    return Total_Score,List_of_Scores


if __name__=="__main__":
    OA_filepath = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Bris_Codes_with_Weights_and_Coords_NEW.json'
    with open(OA_filepath, 'r') as file:
        import json
        Population_Dict = json.load(file)
    
    Cluster_filepath = '/Users/huwtebbutt/Documents/MDM3/3rd_term/commuter_flows.csv'
    import pandas as pd
    df =pd.read_csv(Cluster_filepath)
    columns=df.columns
    Clusters=set(list(df[columns[0]])+list(df[columns[1]]))        
    Stations=[]
    for Cluster in Clusters:
        coords=Population_Dict[Cluster][1]
        Stations.append(coords)
    Score,List=Distance_Score(Population_Dict,Stations)
    
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(dpi=500)
    ax.hist(List)
    ax.set_title('Distribution of distance based scores')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    




        
