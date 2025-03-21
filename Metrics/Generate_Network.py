#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:34:02 2025

@author: huwtebbutt
"""

def Generate_Network(CSV):
    import pandas as pd
    import numpy as np
    import networkx as nx
    df =pd.read_csv(CSV)
    columns=df.columns
    Headers=list(set(list(df[columns[0]])+list(df[columns[1]])))
    
    Matrix=np.zeros((len(Headers),len(Headers)))
    
    with open(CSV,'r',newline='') as file:
        import csv
        file=csv.reader(file)
        for counter,row in enumerate(file):
            if counter==0:
                continue
            i=Headers.index(row[0])
            j=Headers.index(row[1])
            Matrix[i][j]=row[2]
    # Create an empty graph
    G = nx.Graph()
    Node_Labels={}
    for i in range(len(Headers)):
        Node_Labels[i]=Headers[i]
    # Get the number of nodes
    num_nodes = Matrix.shape[0]
    
    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))
    
    # Add edges to the graph
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Matrix[i, j] != 0:  # Assuming 0 means no edge
                G.add_edge(i, j, weight=Matrix[i, j])
    return G,Node_Labels

if __name__=='__main__':
    CSV_File='/Users/huwtebbutt/Documents/MDM3/3rd_term/commuter_flows.csv'
    Network,Node_Labels=Generate_Network(CSV_File)
    
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))

    import networkx as nx
    nx.draw(Network, labels=Node_Labels, node_color='skyblue', node_size=100, edge_color='gray', font_size=7, font_color='black')#
    plt.title('Network Graph')
    plt.show()
                
                
                
            