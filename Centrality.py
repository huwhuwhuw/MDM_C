import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import pickle
'''
Calculates CLoseness centralities for OAs (or MSOAs if csv changed).
'''

def CSV_To_ADJ(data_path):
    """Converts the 2011 (MS)OA raw data table into a weighted adjacency matrix"""

    # import the data and name columns properly
    raw_data = pd.read_csv(OA_data, names = ["Workplace", "Home", "Count"])

    
    # Ensure the count column is numeric
    raw_data["Count"] = pd.to_numeric(raw_data["Count"], errors="coerce")
    
    # aggregate duplicates and transform to matrix
    aggregated_data = raw_data.groupby(["Workplace", "Home"], as_index=False).sum()
    adj_matrix = aggregated_data.pivot(index="Home", columns="Workplace", values="Count").fillna(0)

    # raw data has count of people travelling, to make closeness algorithms
    # work invert the count so larger amounts of journeys are shorter 'distances'
    adj_max = adj_matrix.max(axis=None)
    # keep 0 unchanged as those are no edges
    adj_matrix = adj_matrix.map(lambda x: adj_max-x if x!=0 else x)

    return adj_matrix


path = os.path.dirname(__file__)
OA_data = os.path.join(path, 'Csv_Files', 'Bristol_OA_Data.csv')

data = CSV_To_ADJ(OA_data)

print('Adjacency table created successfully')
print(data.info())

# data.to_csv('OA_Weighted_Adj.csv')

# shorten dataset to 20 locations


graph = nx.from_pandas_adjacency(data, create_using=nx.DiGraph)

print('Graph created')

# nx.draw(graph)
# plt.show()

centrality = nx.closeness_centrality(graph, distance='weight')

print('centrality calculated')

# print(centrality)

cent_sort = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1], reverse=True)}

print('Most central areas: ')
print([x for x,y in zip(cent_sort.items(), range(0, 10))])


# saving the centrality measurements
with open('centrality_measure.pkl', 'wb') as file:
    pickle.dump(cent_sort, file)


# centrality_sorted = {k: v for k, v in sorted(centrality.items(), key=lambda item: item[1])}
# print(list(centrality_sorted.items())[:4])


