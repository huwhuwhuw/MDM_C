import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
import contextily as ctx

# Load MSOA lookup CSV
msoa_lookup = pd.read_csv(r"C:\Users\kitcr\Downloads\msoa_lookup.csv")
msoa_lookup.rename(columns={'OA_code': 'MSOA_code'}, inplace=True)

# Load commuting data
commuting_data = pd.read_csv("Formatted_Commuter_Data_Complete.csv")

# Merge home and work locations
commuting_data = commuting_data.merge(msoa_lookup, left_on='Home OA Code', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'home_Latitude', 'Longitude': 'home_Longitude'}, inplace=True)

# Define Bristol center and radius
bristol_center = (51.4545, -2.5879)  # Latitude, Longitude
radius_km = 15  # Filter nodes within this radius

# Function to check if a location is within the Bristol radius
def is_within_radius(lat, lon, center, max_distance_km):
    return geodesic((lat, lon), center).km <= max_distance_km

# Filter nodes within Bristol commuting area
valid_nodes = set(commuting_data['Home OA Code']).union(set(commuting_data['Work OA Code']))
bristol_nodes = msoa_lookup[
    msoa_lookup['MSOA_code'].isin(valid_nodes) &
    msoa_lookup.apply(lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)
]

# Create graph
G = nx.DiGraph()
for idx, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

# Compute node size based on arrivals and departures
node_sizes_arrival = commuting_data.groupby('Work OA Code')['Commuter Count'].sum().rename("total_arrivals")
node_sizes_departure = commuting_data.groupby('Home OA Code')['Commuter Count'].sum().rename("total_departures")

# Merge node positions into GeoDataFrame
node_positions = nx.get_node_attributes(G, 'pos')
node_gdf = gpd.GeoDataFrame(
    {'MSOA_code': list(node_positions.keys())},
    geometry=gpd.points_from_xy([pos[0] for pos in node_positions.values()],
                                [pos[1] for pos in node_positions.values()]),
    crs="EPSG:4326"
)

# Merge arrival and departure data
node_gdf = node_gdf.merge(node_sizes_arrival, left_on='MSOA_code', right_index=True, how='left')
node_gdf = node_gdf.merge(node_sizes_departure, left_on='MSOA_code', right_index=True, how='left')
node_gdf[['total_arrivals', 'total_departures']] = node_gdf[['total_arrivals', 'total_departures']].fillna(0)

# Convert to Web Mercator for clustering calculations
node_gdf = node_gdf.to_crs(epsg=3857)
# Extract relevant features for clustering
features = node_gdf[['geometry', 'total_arrivals', 'total_departures']].copy()
features['Longitude'] = features['geometry'].x
features['Latitude'] = features['geometry'].y

features['total_arrivals'] = features['total_arrivals'] ** 1.5
features['total_departures'] = features['total_departures'] ** 1.5

# Define clustering input
X = features[['Longitude', 'Latitude', 'total_arrivals', 'total_departures']].values

# Define number of clusters
n_clusters = 60

# Apply K-Medoids Clustering
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
features['KMedoids_Cluster'] = kmedoids.fit_predict(X)

# Compute cluster centroids as the medoids themselves
medoid_indices = kmedoids.medoid_indices_
kmedoids_centroids = features.iloc[medoid_indices][['Longitude', 'Latitude']]

# Convert cluster centers to GeoDataFrame
kmedoids_centroids_gdf = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(kmedoids_centroids['Longitude'], kmedoids_centroids['Latitude'])],
    crs="EPSG:3857"
)
kmedoids_centroids_gdf = gpd.sjoin(kmedoids_centroids_gdf, node_gdf[['MSOA_code', 'geometry']], how="left", predicate="intersects")
kmedoids_centroids_gdf.rename(columns={"MSOA_code": "OA_Code"}, inplace=True)
print(kmedoids_centroids_gdf)

# Compute total commuter flow per cluster
cluster_flows = features.groupby('KMedoids_Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Compute number of nodes in each cluster
cluster_sizes = features.groupby("KMedoids_Cluster").size().rename("Num_Nodes")

# Compute average commuters per node in each cluster
cluster_flows = cluster_flows.merge(cluster_sizes, left_index=True, right_index=True)
cluster_flows["Avg_Flow_Per_Node"] = cluster_flows["Total_Commuters"]

# Rank clusters by Avg Flow Per Node
top_clusters = cluster_flows.nlargest(40, "Avg_Flow_Per_Node").index.tolist()

# Assign "Major" to stops in top clusters, "Minor" otherwise
features["Stop_Type"] = features["KMedoids_Cluster"].apply(lambda cluster: "Major" if cluster in top_clusters else "Minor")

# Update centroids dataframe with stop types
kmedoids_centroids_gdf = kmedoids_centroids_gdf.merge(
    features.groupby("KMedoids_Cluster")["Stop_Type"].first(), left_index=True, right_index=True, how="left"
)

# Reproject for mapping
node_gdf = node_gdf.to_crs(epsg=3857)
kmedoids_centroids_gdf = kmedoids_centroids_gdf.to_crs(epsg=3857)

# Plot results
fig, ax = plt.subplots(figsize=(12, 8))
node_gdf.plot(ax=ax, markersize=2, color="lightgrey", alpha=0.6)
#features.plot(ax=ax, column='KMedoids_Cluster', cmap='tab10', markersize=2)
kmedoids_centroids_gdf[kmedoids_centroids_gdf['Stop_Type'] == 'Major'].plot(ax=ax, color="red", markersize=80, marker="X")
kmedoids_centroids_gdf[kmedoids_centroids_gdf['Stop_Type'] == 'Minor'].plot(ax=ax, color="red", markersize=80, marker="X")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
#ax.set_title("Potential Tram Stop Locations in Bristol (K-Medoids Clustering, k = 40)")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
plt.legend()
plt.show()

# Extract the OA codes of the plotted representative nodes
plotted_oa_codes = kmedoids_centroids_gdf['OA_Code']
plotted_oa_codes.to_csv("kmedoids_oa_codes_60.csv", index=False, header=["OA_Code"])
#print("✅ CSV file saved as 'markov_oa_codes.csv' with", len(plotted_oa_codes), "OA codes.")
