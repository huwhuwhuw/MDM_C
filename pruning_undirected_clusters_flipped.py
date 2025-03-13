import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import matplotlib
import folium
import numpy as np
matplotlib.use('TkAgg')
'''
Creates undirected graph from Clusters made from OA commute data (Adds outgoing commute count and incoming), 
measures edge betweeness centrality and 1/3 of edges with lowest centrality then prunes nodes based on their degree
 and saves a map of the result.
'''
# **Step 1: Load Data**
oa_lookup = pd.read_csv("oa_lookup.csv")

commuting_data = pd.read_csv("Cluster_flows.csv", index_col=0)
commuting_data.reset_index(inplace=True)

# Merge to get home cluster coordinates
commuting_data = commuting_data.merge(
    oa_lookup,
    left_on='Home Cluster OA Code',
    right_on='OA_code',
    how='left'
).rename(columns={'Latitude': 'home latitude', 'Longitude': 'home longitude'})

# Merge to get work cluster coordinates
commuting_data = commuting_data.merge(
    oa_lookup,
    left_on='Work Cluster OA Code',
    right_on='OA_code',
    how='left'
).rename(columns={'Latitude': 'work latitude', 'Longitude': 'work longitude'})

commuting_data = commuting_data.drop(columns=['OA_code_x', 'OA_code_y'])
commuting_data = commuting_data.rename(columns={'Home Cluster OA Code': 'home_OA', 'Work Cluster OA Code': 'work_OA'})

# **Step 2: Define Bristol Area**
bristol_center = (51.4545, -2.5879)
radius_km = 12

def is_within_radius(lat, lon, center, max_distance_km):
    return geodesic((lat, lon), center).km <= max_distance_km

valid_nodes = set(commuting_data['home_OA']).union(set(commuting_data['work_OA']))
bristol_nodes = oa_lookup[
    oa_lookup['OA_code'].isin(valid_nodes) &
    oa_lookup.apply(lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)
]

# **Step 3: Create Undirected Graph**
G = nx.Graph()

# Add nodes with positions
for _, row in bristol_nodes.iterrows():
    G.add_node(row['OA_code'], pos=(row['Longitude'], row['Latitude']))

tidy_commuting_data = commuting_data[commuting_data['home_OA'] != commuting_data['work_OA']]
tidy_commuting_data = tidy_commuting_data[[col for col in ['home_OA', 'work_OA', 'Commuter Count', 'home longitude', 'home latitude', 'work longitude', 'work latitude']]]
pd.set_option("display.max_columns", None)

# Combine weights for undirected edges
edge_weights = {}
for _, row in tidy_commuting_data.iterrows():
    if row['home_OA'] in G.nodes and row['work_OA'] in G.nodes:
        edge = tuple(sorted((row['home_OA'], row['work_OA'])))
        edge_weights[edge] = edge_weights.get(edge, 0) + row['Commuter Count']

# Add edges to the graph with combined weights
for edge, weight in edge_weights.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# **Step 4: Plot the Original Graph**
plt.figure(figsize=(12, 12))
pos = nx.get_node_attributes(G, 'pos')

nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.6)

plt.title('Original Graph: Undirected Commuter Flows in Bristol Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Step 5: Compute and Plot Edge Betweenness Centrality**
edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')

plt.figure(figsize=(8, 6))
plt.hist(edge_betweenness.values(), bins=20, color='green', alpha=0.7)
plt.title('Edge Betweenness Centrality Distribution')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

# **Step 6: Identify and Remove Edges in the Lowest Bin**
counts, bin_edges = np.histogram(list(edge_betweenness.values()), bins=20)
# lowest_bin_edges = [edge for edge, bc in edge_betweenness.items() if bc < bin_edges[8]]
average_edge_betweenness = sum(edge_betweenness.values()) / len(edge_betweenness)

lowest_bin_edges = [edge for edge, bc in edge_betweenness.items() if bc < average_edge_betweenness]

G_reduced = G.copy()
G_reduced.remove_edges_from(lowest_bin_edges)

# **Step 7: Plot the Reduced Graph with Edges Removed**
plt.figure(figsize=(12, 12))
pos_reduced = nx.get_node_attributes(G_reduced, 'pos')

nx.draw_networkx_nodes(G_reduced, pos_reduced, node_size=50, node_color='red', alpha=0.8)
nx.draw_networkx_edges(G_reduced, pos_reduced, edge_color='gray', width=0.5, alpha=0.6)
nx.draw_networkx_labels(G_reduced, pos_reduced, font_size=8, font_color='black')

plt.title('Reduced Graph: Low Betweenness Edges Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Step 8: Compute and Plot Node Degree Centrality in Reduced Graph**
node_degree = nx.degree_centrality(G_reduced)

plt.figure(figsize=(8, 6))
plt.hist(node_degree.values(), bins=20, color='blue', alpha=0.7)
plt.title('Node Degree Centrality Distribution (Reduced Graph)')
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')

# **Step 9: Identify and Remove Nodes in the Lowest Bin**
counts, bin_edges = np.histogram(list(node_degree.values()), bins=100)
# average_node_degree = sum(node_degree.values()) / len(node_degree)
# lowest_bin_nodes = [node for node, dc in node_degree.items() if dc < average_node_degree]
sorted_nodes = sorted(node_degree, key=lambda x: node_degree[x])
num_nodes_to_remove = len(sorted_nodes) // 3
lowest_bin_nodes = sorted_nodes[:num_nodes_to_remove]

G_further_reduced = G_reduced.copy()
G_further_reduced.remove_nodes_from(lowest_bin_nodes)

# **Step 10: Plot the Further Reduced Graph**
plt.figure(figsize=(12, 12))
pos_further_reduced = nx.get_node_attributes(G_further_reduced, 'pos')

nx.draw_networkx_nodes(G_further_reduced, pos_further_reduced, node_size=50, node_color='purple', alpha=0.8)
nx.draw_networkx_edges(G_further_reduced, pos_further_reduced, edge_color='gray', width=0.5, alpha=0.6)
nx.draw_networkx_labels(G_further_reduced, pos_further_reduced, font_size=8, font_color='black')

plt.title('Further Reduced Graph: Low Degree Nodes Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Plot on an Interactive Folium Map**
bristol_map = folium.Map(location=bristol_center, zoom_start=12, tiles="OpenStreetMap")

# **Plot Reduced Graph Edges**
for edge in G_further_reduced.edges:
    loc1 = G_reduced.nodes[edge[0]]['pos'][1], G_reduced.nodes[edge[0]]['pos'][0]  # (lat, lon)
    loc2 = G_reduced.nodes[edge[1]]['pos'][1], G_reduced.nodes[edge[1]]['pos'][0]
    folium.PolyLine([loc1, loc2], color="gray", weight=1.5, opacity=0.5).add_to(bristol_map)

# **Add Reduced Graph Nodes**
for node, data in G_reduced.nodes(data=True):
    folium.CircleMarker(
        location=(data['pos'][1], data['pos'][0]),
        radius=3,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"Node: {node}"
    ).add_to(bristol_map)

# **Save and Display Map**
bristol_map.save("flipped_commuting_reduced_map.html")
print("✅ Interactive map saved as 'flipped_commuting_reduced_map.html'. Open this file in a browser to view.")

plt.show()

'''
to do: use weights of edges instead of betweeness.
'''