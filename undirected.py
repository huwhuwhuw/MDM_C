import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

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
G = nx.Graph()  # Use an undirected graph

# Add nodes with positions
for _, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

tidy_commuting_data = commuting_data[commuting_data['home_MSOA'] != commuting_data['work_MSOA']]
# Combine weights for undirected edges
edge_weights = {}
for _, row in tidy_commuting_data.iterrows():
    if row['home_MSOA'] in G.nodes and row['work_MSOA'] in G.nodes:
        # Create a sorted tuple to represent the undirected edge
        edge = tuple(sorted((row['home_MSOA'], row['work_MSOA'])))
        if edge in edge_weights:
            edge_weights[edge] += row['commuter_count']
        else:
            edge_weights[edge] = row['commuter_count']

# Add edges to the graph with combined weights
for edge, weight in edge_weights.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# **Step 5: Plot the Original Graph**
plt.figure(figsize=(12, 12))

# Get node positions
pos = nx.get_node_attributes(G, 'pos')

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.8)

# Draw edges with weights
edges = G.edges(data=True)
edge_widths = [data['weight'] / 100 for _, _, data in edges]  # Scale edge widths for visualization
nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, alpha=0.6)

# Draw labels (optional, can be commented out if too cluttered)
# nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

# Set title and axis labels
plt.title('Original Graph: Undirected Commuter Flows in Bristol Area')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show plot
coordinates = nx.get_node_attributes(G, 'pos')
# **Step 6: Find Nodes with Largest and Smallest Latitudes and Longitudes**
# Find keys for min/max longitude (first element in tuple)
min_longitude_key = min(coordinates, key=lambda k: coordinates[k][1])
max_longitude_key = max(coordinates, key=lambda k: coordinates[k][1])

# Find keys for min/max latitude (second element in tuple)
min_latitude_key = min(coordinates, key=lambda k: coordinates[k][0])
max_latitude_key = max(coordinates, key=lambda k: coordinates[k][0])

# Print coordinates of the extreme points
print("Min Latitude:", coordinates[min_latitude_key])
print("Max Latitude:", coordinates[max_latitude_key])
print("Min Longitude:", coordinates[min_longitude_key])
print("Max Longitude:", coordinates[max_longitude_key])

# **Step 7: Calculate Betweenness Centrality**
def calculate_node_betweenness(graph):
    """Calculate node betweenness centrality."""
    return nx.betweenness_centrality(graph, weight='weight')

def calculate_edge_betweenness(graph):
    """Calculate edge betweenness centrality."""
    return nx.edge_betweenness_centrality(graph, weight='weight')

# Compute node and edge betweenness centrality
node_betweenness = calculate_node_betweenness(G)
edge_betweenness = calculate_edge_betweenness(G)

# **Step 8: Plot Distributions of Betweenness Centrality**
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot node betweenness centrality distribution
node_bc_values = list(node_betweenness.values())
ax1.hist(node_bc_values, bins=20, color='blue', alpha=0.7)
ax1.set_title('Node Betweenness Centrality Distribution')
ax1.set_xlabel('Betweenness Centrality')
ax1.set_ylabel('Frequency')

# Plot edge betweenness centrality distribution
edge_bc_values = list(edge_betweenness.values())
ax2.hist(edge_bc_values, bins=20, color='green', alpha=0.7)
ax2.set_title('Edge Betweenness Centrality Distribution')
ax2.set_xlabel('Betweenness Centrality')
ax2.set_ylabel('Frequency')

# Show plots
plt.tight_layout()


# **Step 9: Identify Nodes in the Lowest Histogram Bin**
# Get the histogram bins and counts
counts, bin_edges = np.histogram(node_bc_values, bins=20)

# Identify the nodes in the lowest bin
lowest_bin_nodes = [node for node, bc in node_betweenness.items() if bc < bin_edges[1]]

# **Step 10: Remove Nodes in the Lowest Bin and Their Edges**
G_reduced = G.copy()
G_reduced.remove_nodes_from(lowest_bin_nodes)

# **Step 11: Plot the Reduced Graph**
plt.figure(figsize=(12, 12))

# Get node positions for the reduced graph
pos_reduced = nx.get_node_attributes(G_reduced, 'pos')

# Draw nodes
nx.draw_networkx_nodes(G_reduced, pos_reduced, node_size=50, node_color='blue', alpha=0.8)

# Draw edges with weights
edges_reduced = G_reduced.edges(data=True)
edge_widths_reduced = [data['weight'] / 100 for _, _, data in edges_reduced]  # Scale edge widths for visualization
nx.draw_networkx_edges(G_reduced, pos_reduced, edge_color='gray', width=edge_widths_reduced, alpha=0.6)
nx.draw_networkx_labels(G_reduced, pos, font_size=8, font_color='black')

# Draw labels (optional, can be commented out if too cluttered)
# nx.draw_networkx_labels(G_reduced, pos_reduced, font_size=8, font_color='black')

# Set title and axis labels
plt.title('Reduced Graph: Nodes in Lowest Betweenness Bin Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show plot
plt.show()



