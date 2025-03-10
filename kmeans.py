import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from IPython.display import display
import contextily as ctx

# Load MSOA lookup CSV
msoa_lookup = pd.read_csv(r"C:\Users\kitcr\Downloads\MDM3 Trams\Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-8386444323138516297.csv")
msoa_lookup.rename(columns={'MSOA21CD': 'MSOA_code', 'LAT': 'Latitude', 'LONG': 'Longitude'}, inplace=True)

# Load commuting data
commuting_data = pd.read_csv(r"C:\Users\kitcr\Downloads\MDM3 Trams\Bristol_2011_MSOA_data_complete_formatted.csv")

# Merge home and work locations
commuting_data = commuting_data.merge(msoa_lookup, left_on='home_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'home_Latitude', 'Longitude': 'home_Longitude'}, inplace=True)

commuting_data = commuting_data.merge(msoa_lookup, left_on='work_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'work_Latitude', 'Longitude': 'work_Longitude'}, inplace=True)

# Define Bristol center and radius
bristol_center = (51.4545, -2.5879)  # Latitude, Longitude
radius_km = 10  # Filter nodes within this radius

# Function to check if a location is within the Bristol radius
def is_within_radius(lat, lon, center, max_distance_km):
    return geodesic((lat, lon), center).km <= max_distance_km

# Filter nodes within Bristol commuting area
valid_nodes = set(commuting_data['home_MSOA']).union(set(commuting_data['work_MSOA']))
bristol_nodes = msoa_lookup[
    msoa_lookup['MSOA_code'].isin(valid_nodes) &
    msoa_lookup.apply(lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)
]


# Create graph
G = nx.DiGraph()
for idx, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

# Compute node size based on arrivals and departures
node_sizes_arrival = commuting_data.groupby('work_MSOA')['commuter_count'].sum().rename("total_arrivals")
node_sizes_departure = commuting_data.groupby('home_MSOA')['commuter_count'].sum().rename("total_departures")

# Merge node positions into GeoDataFrame
node_positions = nx.get_node_attributes(G, 'pos')
node_gdf = gpd.GeoDataFrame(
    {'MSOA_code': list(node_positions.keys())},
    geometry=gpd.points_from_xy([pos[0] for pos in node_positions.values()],
                                [pos[1] for pos in node_positions.values()]),
    crs="EPSG:4326"
)

total_msoa = len(node_gdf)
print(f"Total unique MSOAs involved in commuting: {total_msoa}")

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

# Define clustering input
X = features[['Longitude', 'Latitude', 'total_arrivals', 'total_departures']].values

# Define number of clusters
n_clusters = 74

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
features['KMeans_Cluster'] = kmeans.fit_predict(X)

# Compute cluster centroids as the mean of each cluster's locations
kmeans_centroids = features.groupby('KMeans_Cluster')[['Longitude', 'Latitude']].mean().reset_index()

# Convert cluster centers to GeoDataFrame
kmeans_centroids_gdf = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(kmeans_centroids['Longitude'], kmeans_centroids['Latitude'])],
    crs="EPSG:3857"
)

# Compute total commuter flow per cluster
cluster_flows = features.groupby('KMeans_Cluster')[['total_arrivals', 'total_departures']].sum()
cluster_flows['Total_Commuters'] = cluster_flows['total_arrivals'] + cluster_flows['total_departures']

# Compute number of nodes in each cluster
cluster_sizes = features.groupby("KMeans_Cluster").size().rename("Num_Nodes")

# Compute average commuters per node in each cluster
cluster_flows = cluster_flows.merge(cluster_sizes, left_index=True, right_index=True)
cluster_flows["Avg_Flow_Per_Node"] = cluster_flows["Total_Commuters"] / cluster_flows["Num_Nodes"]

# Rank clusters by Avg Flow Per Node
top_clusters = cluster_flows.nlargest(2, "Avg_Flow_Per_Node").index.tolist()

# Assign "Major" to stops in top clusters, "Minor" otherwise
features["Stop_Type"] = features["KMeans_Cluster"].apply(lambda cluster: "Major" if cluster in top_clusters else "Minor")

# Update centroids dataframe with stop types
kmeans_centroids_gdf = kmeans_centroids_gdf.merge(
    features.groupby("KMeans_Cluster")["Stop_Type"].first(), left_index=True, right_index=True, how="left"
)

# Reproject for mapping
node_gdf = node_gdf.to_crs(epsg=3857)
kmeans_centroids_gdf = kmeans_centroids_gdf.to_crs(epsg=3857)

# Plot results
fig, ax = plt.subplots(figsize=(12, 8))
node_gdf.plot(ax=ax, markersize=20, color="lightgrey", alpha=0.6)
features.plot(ax=ax, column='KMeans_Cluster', cmap='tab10', markersize=50)
kmeans_centroids_gdf[kmeans_centroids_gdf['Stop_Type'] == 'Major'].plot(ax=ax, color="black", markersize=80, marker="X")
kmeans_centroids_gdf[kmeans_centroids_gdf['Stop_Type'] == 'Minor'].plot(ax=ax, color="black", markersize=80, marker="X")
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_title("Optimal Tram Station Locations in Bristol (K-Means Clustering)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.show()


