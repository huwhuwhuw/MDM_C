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

# Load OA lookup CSV
oa_lookup = pd.read_csv(r"C:\Users\kitcr\Downloads\oa_lookup.csv")

# Compute node size based on arrivals and departures
node_sizes_arrival = adjacency_matrix.sum(axis=0)
node_sizes_departure = adjacency_matrix.sum(axis=1)

# Merge node positions into GeoDataFrame
node_positions = {code: (row['Longitude'], row['Latitude']) for code, row in
                  oa_lookup.set_index('OA_code').iterrows()}
node_gdf = gpd.GeoDataFrame(
    {'OA_code': list(node_positions.keys())},
    geometry=[Point(pos) for pos in node_positions.values()],
    crs="EPSG:4326"
)

# Merge commuter data into GeoDataFrame
node_gdf['total_arrivals'] = node_gdf['OA_code'].map(node_sizes_arrival)
node_gdf['total_departures'] = node_gdf['OA_code'].map(node_sizes_departure)
node_gdf[['total_arrivals', 'total_departures']] = node_gdf[['total_arrivals', 'total_departures']].fillna(0)

# Convert to Web Mercator
node_gdf = node_gdf.to_crs(epsg=3857)

# Assign clusters to nodes
node_gdf['Cluster'] = node_gdf['OA_code'].map(node_cluster_map)

# Compute commuter flow stats by cluster
cluster_flows = node_gdf.groupby('Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Set thresholds for merging clusters
mean_flow = cluster_flows['Total_Commuters'].mean()
threshold_low = mean_flow * 0.5  # Adjusted threshold
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
distance_threshold = 1000 # Merge stops within 1800 meters

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
num_stops = len(merged_tram_stops_gdf)
print(f"Number of tram stops after merging: {num_stops}")

# Identify the 10 lowest commuter count tram stops
lowest_tram_stops = merged_tram_stops_gdf.nsmallest(27, "total_arrivals")

import matplotlib.pyplot as plt

# ---- PLOTTING ----
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all tram stops in red
merged_tram_stops_gdf.plot(ax=ax, color='red', markersize=80, marker='X', label="Tram Stops")

# Plot the 10 lowest commuter count tram stops in black
lowest_tram_stops.plot(ax=ax, color='black', markersize=80, marker='X', label="Low Traffic Stops")

# Add OA codes as labels above each marker
for idx, row in merged_tram_stops_gdf.iterrows():
    ax.text(
        row.geometry.x, row.geometry.y + 150,  # Adjust y-position slightly for visibility
        str(row["OA_code"]), fontsize=9, color="black", ha='center'
    )

# Add basemap for geographic context
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Formatting
ax.set_title("Optimal Tram Station Locations in Bristol (Markov Clustering)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.show()

import folium
import networkx as nx

# Ensure tram stop locations are extracted correctly
if "OA_code" not in merged_tram_stops_gdf.columns:
    raise ValueError("Column 'OA_code' not found in merged_tram_stops_gdf!")

tram_stops = merged_tram_stops_gdf.set_index("OA_code")

# Calculate total commuter flow (arrivals + departures)
tram_stops["total_commuters"] = tram_stops["total_arrivals"] + tram_stops["total_departures"]

# ✅ Fix: Correctly get the lowest commuter count stops
lowest_tram_stops = tram_stops["total_commuters"].nsmallest(22).index.tolist()

# Define tram lines (edges between stops)
manual_edges = [
    ("E00074370", "E00073742"), # LINE 1
    ("E00073742", "E00073325"),
    ("E00073325", "E00174285"),
    ("E00174285", "E00174050"),
    ("E00174050", "E00073425"),
    ("E00073425", "E00174312"),
    ("E00174312", "E00074104"),
    ("E00074104", "E00073921"),
    ("E00073921", "E00075310"),
    ("E00075310", "E00075339"),
    ("E00075339", "E00075523"),
    ("E00073342", "E00073396"), # LINE 2
    ("E00073396", "E00074002"),
    ("E00074002", "E00174242"),
    ("E00174242", "E00174050"),
    ("E00174242", "E00174312"),
    ("E00075658", "E00073698"), # LINE 3
    ("E00073698", "E00074057"),
    ("E00074057", "E00174312"),
]


# Validate that tram stops exist before creating edges
valid_edges = [(s1, s2) for s1, s2 in manual_edges if s1 in tram_stops.index and s2 in tram_stops.index]

if not valid_edges:
    raise ValueError("No valid edges found! Check that OA codes in manual_edges exist in tram_stops.")

# Create a graph
G_manual = nx.Graph()
G_manual.add_edges_from(valid_edges)

# Ensure CRS is correct
tram_stops = tram_stops.to_crs(epsg=4326)  # Convert to Lat/Lon

# Extract coordinates properly
tram_stops["Latitude"] = tram_stops.geometry.y
tram_stops["Longitude"] = tram_stops.geometry.x

# Ensure coordinates are not NaN
tram_stops = tram_stops.dropna(subset=["Latitude", "Longitude"])

# Verify map centering
map_center = [tram_stops["Latitude"].mean(), tram_stops["Longitude"].mean()]
print(f"Map Center: {map_center}")

# Create a Folium map
m_manual = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

# Add tram stops to the map
for oa_code, row in tram_stops.iterrows():
    color = "black" if oa_code in lowest_tram_stops else "red"  # Black for lowest traffic stops, Red otherwise

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"Stop: {oa_code} (Total Commuters: {row['total_commuters']})",
    ).add_to(m_manual)

# Add manual tram lines
for stop1, stop2 in valid_edges:
    try:
        latlon1 = (tram_stops.loc[stop1, "Latitude"], tram_stops.loc[stop1, "Longitude"])
        latlon2 = (tram_stops.loc[stop2, "Latitude"], tram_stops.loc[stop2, "Longitude"])

        folium.PolyLine([latlon1, latlon2], color="blue", weight=3, opacity=0.7).add_to(m_manual)
    except KeyError:
        print(f"⚠️ Error: One of the stops ({stop1}, {stop2}) is missing from tram_stops!")

# Save the interactive map
manual_map_path = "Custom_Tram_Network.html"
m_manual.save(manual_map_path)

print(f"✅ Custom tram network map saved as {manual_map_path}")


