import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
import markov_clustering as mc
from scipy.spatial import cKDTree
import statistics

# Load adjacency matrix from provided file
file_path = r"C:\Users\kitcr\Downloads\MDM3 Trams\2011_OAs_Bris_GlousNorth_Formatted.csv"
adjacency_matrix = pd.read_csv(file_path, index_col=0)

# Convert to NumPy array
matrix = adjacency_matrix.values

# Apply Markov Clustering
result = mc.run_mcl(matrix, inflation=1.1)
clusters = mc.get_clusters(result)

# Create a mapping of nodes to clusters
node_cluster_map = {}
for cluster_id, cluster in enumerate(clusters):
    for node in cluster:
        node_cluster_map[adjacency_matrix.index[node]] = cluster_id

print(clusters)
print(node_cluster_map)

# Load MSOA lookup CSV
msoa_lookup = pd.read_csv(r"C:\Users\kitcr\Downloads\msoa_lookup.csv")
msoa_lookup.rename(columns={'OA_code': 'MSOA_code'}, inplace=True)

# Compute node size based on arrivals and departures
node_sizes_arrival = adjacency_matrix.sum(axis=0)
node_sizes_departure = adjacency_matrix.sum(axis=1)

# Merge node positions into GeoDataFrame
node_positions = {code: (row['Longitude'], row['Latitude']) for code, row in
                  msoa_lookup.set_index('MSOA_code').iterrows()}
node_gdf = gpd.GeoDataFrame(
    {'MSOA_code': list(node_positions.keys())},
    geometry=[Point(pos) for pos in node_positions.values()],
    crs="EPSG:4326"
)

# Merge commuter data into GeoDataFrame
node_gdf['total_arrivals'] = node_gdf['MSOA_code'].map(node_sizes_arrival)
node_gdf['total_departures'] = node_gdf['MSOA_code'].map(node_sizes_departure)
node_gdf[['total_arrivals', 'total_departures']] = node_gdf[['total_arrivals', 'total_departures']].fillna(0)
node_gdf = node_gdf[~((node_gdf['total_arrivals'] == 0) & (node_gdf['total_departures'] == 0))].reset_index(drop=True)

# Convert to Web Mercator
node_gdf = node_gdf.to_crs(epsg=3857)

# Assign clusters to nodes
node_gdf['Cluster'] = node_gdf['MSOA_code'].map(node_cluster_map)
print(node_gdf[['Cluster']])

# Compute commuter flow stats by cluster
cluster_flows = node_gdf.groupby('Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']
print(cluster_flows)

from scipy.spatial import KDTree

# Define a threshold for low flow (e.g., bottom 40% of total commuter flow)
low_flow_threshold = cluster_flows['Total_Commuters'].quantile(0.40)
low_flow_clusters = cluster_flows[cluster_flows['Total_Commuters'] <= low_flow_threshold].index

# Extract cluster centroids (mean coordinates of nodes in each cluster)
cluster_centroids = node_gdf.groupby('Cluster').geometry.apply(lambda g: g.unary_union.centroid)
centroid_coords = np.array([(p.x, p.y) for p in cluster_centroids])
cluster_ids = np.array(cluster_centroids.index)

# Build a KDTree for fast nearest neighbor lookup
tree = KDTree(centroid_coords)

# Merge low-flow clusters to the nearest high-flow cluster
for cluster in low_flow_clusters:
    cluster_index = np.where(cluster_ids == cluster)[0][0]  # Get index of the cluster
    _, nearest_index = tree.query(centroid_coords[cluster_index], k=5)  # k=2 to exclude itself
    nearest_cluster = cluster_ids[nearest_index[1]]  # Pick the nearest non-itself cluster

    # Reassign all nodes in the low-flow cluster to the nearest high-flow cluster
    node_gdf.loc[node_gdf['Cluster'] == cluster, 'Cluster'] = nearest_cluster

print("✅ Low-flow clusters merged successfully.")
print("🔍 Unique clusters after merging:", node_gdf['Cluster'].nunique())

# Find the OA with the highest total flow in each cluster
# Calculate total flow per OA
node_gdf['total_flow'] = node_gdf['total_arrivals'] + node_gdf['total_departures']
representative_nodes = node_gdf.loc[node_gdf.groupby('Cluster')['total_flow'].idxmax()].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 8))
node_gdf.plot(ax=ax, markersize=5, color='lightgrey', alpha=0.5)
representative_nodes.plot(ax=ax, color='red', markersize=80, marker='X', label="Tram Stop Locations")

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

ax.set_title("Top-Flow OA in Each Markov Cluster")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.show()

# Extract the OA codes of the plotted representative nodes
plotted_oa_codes = representative_nodes['MSOA_code']
plotted_oa_codes.to_csv("markov_oa_codes_merged.csv", index=False, header=["OA_code"])
print("✅ CSV file saved as 'markov_oa_codes.csv' with", len(plotted_oa_codes), "OA codes.")



