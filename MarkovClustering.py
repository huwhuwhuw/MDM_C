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
result = mc.run_mcl(matrix, inflation=1.8)
clusters = mc.get_clusters(result)

# Create a mapping of nodes to clusters
node_cluster_map = {}
for cluster_id, cluster in enumerate(clusters):
    for node in cluster:
        node_cluster_map[adjacency_matrix.index[node]] = cluster_id

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

# Convert to Web Mercator
node_gdf = node_gdf.to_crs(epsg=3857)

# Assign clusters to nodes
node_gdf['Cluster'] = node_gdf['MSOA_code'].map(node_cluster_map)

# Compute commuter flow stats by cluster
cluster_flows = node_gdf.groupby('Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Set thresholds for merging clusters
mean_flow = cluster_flows['Total_Commuters'].mean()
threshold_low = mean_flow
#threshold_high = mean_flow * 1.5
# Set threshold for low-traffic clusters
mean_flow = cluster_flows['Total_Commuters'].mean()
median_flow = statistics.median(cluster_flows['Total_Commuters'])
threshold_low = mean_flow * 0.5
low_traffic_clusters = cluster_flows[cluster_flows['Total_Commuters'] < threshold_low].index.tolist()
# Merge low-traffic clusters into nearest high-traffic cluster
high_traffic_clusters = cluster_flows[cluster_flows['Total_Commuters'] >= threshold_low].index.tolist()

# Create spatial tree for efficient nearest neighbor search
high_traffic_centroids = node_gdf[node_gdf['Cluster'].isin(high_traffic_clusters)][['geometry']]
tree = cKDTree(np.array(list(zip(high_traffic_centroids.geometry.x, high_traffic_centroids.geometry.y))))

# Reassign low-traffic clusters
for cluster in low_traffic_clusters:
    nodes = node_gdf[node_gdf['Cluster'] == cluster]
    for idx, row in nodes.iterrows():
        dist, nearest_idx = tree.query([row.geometry.x, row.geometry.y])
        nearest_cluster = node_gdf.iloc[nearest_idx]['Cluster']
        node_gdf.at[idx, 'Cluster'] = nearest_cluster

# Recompute cluster stats after merging
cluster_flows = node_gdf.groupby('Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Select the highest traffic node within each cluster as the tram stop
tram_stops = node_gdf.loc[node_gdf.groupby('Cluster')['total_arrivals'].idxmax()]
# Identify clusters above the high threshold
high_traffic_clusters = cluster_flows[cluster_flows['Total_Commuters'] >= threshold_low].index.tolist()

# Select the highest traffic node within each cluster as the tram stop
tram_stops = node_gdf.loc[node_gdf.groupby('Cluster')['total_arrivals'].idxmax()]

# Merge all tram stops within the given distance threshold
distance_threshold = 1500  # Merge stops within 500 meters

# Build a spatial tree for tram stops
tram_stop_tree = cKDTree(np.array(list(zip(tram_stops.geometry.x, tram_stops.geometry.y))))

# List to store merged tram stops
merged_stops = []
assigned_clusters = {}

# Iterate through all tram stops
for idx, row in tram_stops.iterrows():
    if row['Cluster'] in assigned_clusters:
        continue  # Skip already merged stops

    # Query all nearby stops within the distance threshold
    dists, neighbors = tram_stop_tree.query([row.geometry.x, row.geometry.y], k=len(tram_stops),
                                            distance_upper_bound=distance_threshold)

    # Filter neighbors within the distance threshold
    valid_neighbors = [tram_stops.iloc[n_idx] for i, n_idx in enumerate(neighbors) if
                       dists[i] < distance_threshold and n_idx != idx]

    if valid_neighbors:
        # Merge into the stop with the highest arrivals in the group
        valid_neighbors.append(row)
        best_stop = max(valid_neighbors, key=lambda x: x['total_arrivals'])
        merged_stops.append(best_stop)

        # Assign merged cluster to all neighbors
        for neighbor in valid_neighbors:
            assigned_clusters[neighbor['Cluster']] = best_stop['Cluster']
    else:
        # If no neighbors, keep this stop
        merged_stops.append(row)

# Reassign clusters based on merging
for old_cluster, new_cluster in assigned_clusters.items():
    node_gdf.loc[node_gdf['Cluster'] == old_cluster, 'Cluster'] = new_cluster

# Recompute cluster stats after merging
cluster_flows = node_gdf.groupby('Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Convert merged stops to GeoDataFrame
merged_tram_stops_gdf = gpd.GeoDataFrame(merged_stops, geometry='geometry', crs=node_gdf.crs)

# Print number of stops
num_stops = len(merged_tram_stops_gdf)
print(f"Number of tram stops after merging: {num_stops}")

# Plot results
fig, ax = plt.subplots(figsize=(12, 8))
node_gdf.plot(ax=ax, markersize=2, color='lightgrey', alpha=0.6)
merged_tram_stops_gdf.plot(ax=ax, color='red', markersize=80, marker='X')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title("Optimal Tram Station Locations in Bristol (Markov Clustering)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.show()

# Create a mapping of clusters to tram stop OA codes
cluster_to_tram_stop = merged_tram_stops_gdf.set_index('Cluster')['MSOA_code'].to_dict()

# Create a DataFrame for commuter flows between clusters
flows = []
for i, row in adjacency_matrix.iterrows():
    if i in node_gdf['MSOA_code'].values:
        # Get the home cluster using tram stop codes
        home_cluster = node_gdf.loc[node_gdf['MSOA_code'] == i, 'Cluster'].values[0]
        home_tram_stop = cluster_to_tram_stop.get(home_cluster, None)

        for j, commuter_count in row.items():
            if commuter_count > 0 and j in node_gdf['MSOA_code'].values:
                # Get the work cluster using tram stop codes
                work_cluster = node_gdf.loc[node_gdf['MSOA_code'] == j, 'Cluster'].values[0]
                work_tram_stop = cluster_to_tram_stop.get(work_cluster, None)

                if home_tram_stop is not None and work_tram_stop is not None:
                    flows.append([home_tram_stop, work_tram_stop, commuter_count])

# Convert to DataFrame
flow_df = pd.DataFrame(flows, columns=['Home Cluster OA Code', 'Work Cluster OA Code', 'Commuter Count'])

# Remove duplicates by grouping and summing commuter counts
flow_df = flow_df.groupby(['Home Cluster OA Code', 'Work Cluster OA Code'], as_index=False).sum()

# Save to CSV
output_path = r"C:\Users\kitcr\Downloads\commuter_flows.csv"
flow_df.to_csv(output_path, index=False)

print(f"Commuter flow data saved to {output_path}")

