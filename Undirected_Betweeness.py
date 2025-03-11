import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import matplotlib
import folium
import numpy as np
matplotlib.use('TkAgg')
'''
Creates undirected graph from MSOA commute data (Adds outgoing commute count and incoming), 
measures betweeness centrality for edges and nodes and deletes nodes with lowest betweeness centrality 
(lowest histogram bin in plot), further deletes non-central edges and saves a map of the result.
'''
# **Step 1: Load Data**
msoa_lookup = pd.read_csv("MSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_2022_-5777602578195197657.csv")
msoa_lookup.rename(columns={'MSOA11CD': 'MSOA_code', 'LAT': 'Latitude', 'LONG': 'Longitude'}, inplace=True)

commuting_data = pd.read_csv("Commute_Table_Modified.csv", index_col=0)
commuting_data.reset_index(inplace=True)
commuting_data.rename(columns={'index': 'home_MSOA'}, inplace=True)

# Convert to long-form edge list
commuting_data = commuting_data.melt(id_vars=['home_MSOA'], var_name='work_MSOA', value_name='commuter_count')
commuting_data = commuting_data[commuting_data['commuter_count'] > 0]

# **Step 2: Merge Home and Work Locations**
commuting_data = commuting_data.merge(msoa_lookup, left_on='home_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'home_Latitude', 'Longitude': 'home_Longitude'}, inplace=True)

commuting_data = commuting_data.merge(msoa_lookup, left_on='work_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'work_Latitude', 'Longitude': 'work_Longitude'}, inplace=True)

# **Step 3: Define Bristol Area**
bristol_center = (51.4545, -2.5879)
radius_km = 9

def is_within_radius(lat, lon, center, max_distance_km):
    return geodesic((lat, lon), center).km <= max_distance_km

valid_nodes = set(commuting_data['home_MSOA']).union(set(commuting_data['work_MSOA']))
bristol_nodes = msoa_lookup[
    msoa_lookup['MSOA_code'].isin(valid_nodes) &
    msoa_lookup.apply(lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)
]

# **Step 4: Create Undirected Graph**
G = nx.Graph()

# Add nodes with positions
for _, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

tidy_commuting_data = commuting_data[commuting_data['home_MSOA'] != commuting_data['work_MSOA']]

# Combine weights for undirected edges
edge_weights = {}
for _, row in tidy_commuting_data.iterrows():
    if row['home_MSOA'] in G.nodes and row['work_MSOA'] in G.nodes:
        edge = tuple(sorted((row['home_MSOA'], row['work_MSOA'])))
        edge_weights[edge] = edge_weights.get(edge, 0) + row['commuter_count']

# Add edges to the graph with combined weights
for edge, weight in edge_weights.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# **Step 5: Plot the Original Graph**
plt.figure(figsize=(12, 12))
pos = nx.get_node_attributes(G, 'pos')

nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.6)

plt.title('Original Graph: Undirected Commuter Flows in Bristol Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Step 6: Compute and Plot Node Betweenness Centrality**
node_betweenness = nx.betweenness_centrality(G, weight='weight')

plt.figure(figsize=(8, 6))
plt.hist(node_betweenness.values(), bins=20, color='blue', alpha=0.7)
plt.title('Node Betweenness Centrality Distribution')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

# **Step 7: Identify and Remove Nodes in the Lowest Bin**
counts, bin_edges = np.histogram(list(node_betweenness.values()), bins=20)
lowest_bin_nodes = [node for node, bc in node_betweenness.items() if bc < bin_edges[1]]

G_reduced = G.copy()
G_reduced.remove_nodes_from(lowest_bin_nodes)

# **Step 8: Plot the Reduced Graph with MSOA Codes**
plt.figure(figsize=(12, 12))
pos_reduced = nx.get_node_attributes(G_reduced, 'pos')

nx.draw_networkx_nodes(G_reduced, pos_reduced, node_size=50, node_color='red', alpha=0.8)
nx.draw_networkx_edges(G_reduced, pos_reduced, edge_color='gray', width=0.5, alpha=0.6)
nx.draw_networkx_labels(G_reduced, pos_reduced, font_size=8, font_color='black')

plt.title('Reduced Graph: Nodes in Lowest Betweenness Bin Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Step 9: Compute and Plot Edge Betweenness Centrality in Reduced Graph**
edge_betweenness_reduced = nx.edge_betweenness_centrality(G_reduced, weight='weight')

plt.figure(figsize=(8, 6))
plt.hist(edge_betweenness_reduced.values(), bins=20, color='green', alpha=0.7)
plt.title('Edge Betweenness Centrality Distribution (Reduced Graph)')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

# **Step 10: Identify and Remove Edges in the Lowest Bin**
counts, bin_edges = np.histogram(list(edge_betweenness_reduced.values()), bins=20)
lowest_bin_edges = [edge for edge, bc in edge_betweenness_reduced.items() if bc < bin_edges[7]]

G_further_reduced = G_reduced.copy()
G_further_reduced.remove_edges_from(lowest_bin_edges)

# **Step 11: Plot the Further Reduced Graph**
plt.figure(figsize=(12, 12))
pos_further_reduced = nx.get_node_attributes(G_further_reduced, 'pos')

nx.draw_networkx_nodes(G_further_reduced, pos_further_reduced, node_size=50, node_color='purple', alpha=0.8)
nx.draw_networkx_edges(G_further_reduced, pos_further_reduced, edge_color='gray', width=0.5, alpha=0.6)
nx.draw_networkx_labels(G_further_reduced, pos_further_reduced, font_size=8, font_color='black')

plt.title('Further Reduced Graph: Low Betweenness Edges Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# **Find Extreme Nodes (North, South, East, West)**
northernmost = max(G_reduced.nodes, key=lambda n: G_reduced.nodes[n]['pos'][1])
southernmost = min(G_reduced.nodes, key=lambda n: G_reduced.nodes[n]['pos'][1])
easternmost = max(G_reduced.nodes, key=lambda n: G_reduced.nodes[n]['pos'][0])
westernmost = min(G_reduced.nodes, key=lambda n: G_reduced.nodes[n]['pos'][0])

# **Print Extreme Node Coordinates**
print("Extreme Nodes and Their Coordinates:")
print(f"📍 Northernmost Node: {northernmost}, Coordinates: {G_reduced.nodes[northernmost]['pos']}")
print(f"📍 Southernmost Node: {southernmost}, Coordinates: {G_reduced.nodes[southernmost]['pos']}")
print(f"📍 Easternmost Node: {easternmost}, Coordinates: {G_reduced.nodes[easternmost]['pos']}")
print(f"📍 Westernmost Node: {westernmost}, Coordinates: {G_reduced.nodes[westernmost]['pos']}")

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
bristol_map.save("bristol_commuting_reduced_map.html")
print("✅ Interactive map saved as 'bristol_commuting_reduced_map.html'. Open this file in a browser to view.")

plt.show()