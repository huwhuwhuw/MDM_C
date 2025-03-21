#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:12:55 2025

@author: huwtebbutt
"""
#Problems, if a person lives near there work, they will give a high score, when they probably wouldnt use a tram
#Skips people if there is no station near them, i could make it where it gives a zero, but thats more about station placemnt than line placement
#


def Score_Network(Line_Network,Node_Labels,Dataframes,Dict,Walkable_Distance=1):
    import networkx as nx
    import pandas as pd
    from geopy.distance import geodesic

    def Drop_Non_Bristol(df,Dict):
        #Codes coresponding to each row
        rows=list(df['currently residing in : 2011 output area'])
        df = df.drop(columns='currently residing in : 2011 output area')
        
        #Codes corresponding to each column
        columns=list(df.columns)
        
        #List of all codes
        both=list(set(rows+columns))
        for code in both:
            #Remove non bristol codes
            if not code in Dict.keys():
                if code in columns:
                    df=df.drop(columns=code)
                if code in rows:
                    i=rows.index(code)
                    df=df.drop(index=i)
        #I hate pandas
        #Pandas stores column names but not row names, so store manually
        newrows=[]
        for i in df.index:
            newrows.append(rows[i])
        df=df.reset_index(drop=True)
        return df,newrows
    
    def Nearest_Stations(Point,Stations,Coord_Dict,Walkable_Distance=Walkable_Distance):
        #Find each station within a walkable distance of a point
        #Walkable distance is set to 1km which is around 15minutes of walking
        Point_Coords=Coord_Dict[Point][1]
        
        Walkable_Stations=[]
        Diffs=[]
        
        for Station in Stations:
            Station_Coords=Coord_Dict[Station][1]
            Diff=geodesic(Station_Coords,Point_Coords).km
            if Diff < Walkable_Distance:
                Walkable_Stations.append(Station)
                Diffs.append(Diff)
        return Walkable_Stations
    
    def Line_Score(df,df_codes,Network,Node_Labels,Dict):
        Scores=[]
        #Find minimum amount of lines to go from point to point in commuter data
        for i,code in enumerate(df_codes):
            #List of stations within walking distance of point 1
            Stations_1=Nearest_Stations(code,list(Node_Labels.values()),Dict)
            
            row=list(df.loc[i])
            if not Stations_1:
                #If no stations within walking distance give no score
                continue
            for j,item in enumerate(row):
                #If item is 0, then no one commutes to point 2 from point 1
                if item!=0:
                    #Figure out code
                    code=df.columns[j]
                    Stations_2=Nearest_Stations(code,list(Node_Labels.values()),Dict)
                    if not Stations_2:
                        continue
                    
                    #Once Stations have been found near the 2 points
                    #Find the shortest path between any two of them
                    Shortest_paths=[]
                    for Station_1 in Stations_1:
                        Station_1=list(Node_Labels.values()).index(Station_1)
                        for Station_2 in Stations_2:
                            Station_2=list(Node_Labels.values()).index(Station_2)
                            Shortest_path = nx.shortest_path(Network, source=Station_1, target=Station_2)
                            Shortest_paths.append(len(Shortest_path))
                    
                    #Give a score that rewards short paths
                    Score=item/min(Shortest_paths)
                    Scores.append(Score)
        
        return Scores
    
    #If only 1 df is supplied
    if type(Dataframes)==pd.core.frame.DataFrame:
        Dataframes=[Dataframes]
    
    Scores=[]
    for df in Dataframes:
        #Filter dfs
        df,df_codes=Drop_Non_Bristol(df,Dict)
        Scores+=Line_Score(df,df_codes,Line_Network,Node_Labels,Dict)
    #Find average score
    Score=sum(Scores)/len(Scores)
    return Score

if __name__=='__main__':
    
    import pandas as pd
    #Oa commute data, change filepath as needed
    Bath_Som=pd.read_csv('/Users/huwtebbutt/Documents/MDM3/3rd_term/Old_Files/2011_OAs_Bath_NorthSomers_Cleaned.csv')
    Bris_Glou=pd.read_csv('/Users/huwtebbutt/Documents/MDM3/3rd_term/Old_Files/2011_OAs_Bris_GlousNorth_Cleaned.csv')
    
    #Dict of Codes and coords, change filepath as needed
    filepath = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Bris_Codes_with_Weights_and_Coords_NEW.json'
    with open(filepath, 'r') as file:
        import json
        Code_Coords_Dict = json.load(file)
    
    #Network of stops and which are connected to which
    #All stops on a line must be connected to all other stops on a line
    #Stops with multiple lines are connected to all stops on every line
    from Generate_Network import Generate_Network #Example network where every stop is connected to every other stop
    #Replace Network with your own
    CSV_File='/Users/huwtebbutt/Documents/MDM3/3rd_term/commuter_flows.csv'
    Network,Node_Labels=Generate_Network(CSV_File)
        
    
    Score=Score_Network(Network, Node_Labels,Dataframes=[Bath_Som,Bris_Glou],Dict=Code_Coords_Dict)
    
    