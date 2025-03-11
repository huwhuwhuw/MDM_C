import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import contextily as ctx
from geopy.distance import geodesic
'''
PLot network for MSOA 2011 commute data (arrivals and departures)
'''
# Load MSOA lookup CSV
msoa_lookup = pd.read_csv(
    r"C:\Users\kitcr\Downloads\MDM3 Trams\Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-8386444323138516297.csv")
msoa_lookup.rename(columns={'MSOA21CD': 'MSOA_code', 'LAT': 'Latitude', 'LONG': 'Longitude'}, inplace=True)

# Load commuting data
commuting_data = pd.read_csv(r"C:\Users\kitcr\Downloads\MDM3 Trams\Bristol_2011_MSOA_data_complete_formatted.csv")

# Merge home and work locations
commuting_data = commuting_data.merge(msoa_lookup, left_on='home_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'home_Latitude', 'Longitude': 'home_Longitude'}, inplace=True)

commuting_data = commuting_data.merge(msoa_lookup, left_on='work_MSOA', right_on='MSOA_code', how='left')
commuting_data.rename(columns={'Latitude': 'work_Latitude', 'Longitude': 'work_Longitude'}, inplace=True)

# **Define Bristol Center and Radius**
bristol_center = (51.4545, -2.5879)  # Latitude, Longitude
radius_km = 10  # Filter nodes within this radius

# **Function to Check If a Location is Within the Bristol Radius**
def is_within_radius(lat, lon, center, max_distance_km):
    return geodesic((lat, lon), center).km <= max_distance_km

# **Filter Nodes: Only Include Those That Appear in Commuting Data**
valid_nodes = set(commuting_data['home_MSOA']).union(set(commuting_data['work_MSOA']))
bristol_nodes = msoa_lookup[
    msoa_lookup['MSOA_code'].isin(valid_nodes) &
    msoa_lookup.apply(lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)
]

# **Step 1: Create the Directed Graph**
G = nx.DiGraph()

# Add only nodes that appear in the commuting data
for idx, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

# **Step 2: Compute Node Size Based on Arrivals (work_MSOA)**
node_sizes_arrival = commuting_data.groupby('work_MSOA')['commuter_count'].sum().rename("total_arrivals")

# **Step 2b: Compute Node Size Based on Departures (home_MSOA)**
node_sizes_departure = commuting_data.groupby('home_MSOA')['commuter_count'].sum().rename("total_departures")

# **Step 3: Merge Data into Node GeoDataFrame**
node_positions = nx.get_node_attributes(G, 'pos')
node_gdf = gpd.GeoDataFrame(
    {'MSOA_code': list(node_positions.keys())},
    geometry=gpd.points_from_xy([pos[0] for pos in node_positions.values()],
                                [pos[1] for pos in node_positions.values()]),
    crs="EPSG:4326"
)

# Merge arrival and departure data into the node dataframe
node_gdf = node_gdf.merge(node_sizes_arrival, left_on='MSOA_code', right_index=True, how='left')
node_gdf = node_gdf.merge(node_sizes_departure, left_on='MSOA_code', right_index=True, how='left')
node_gdf[['total_arrivals', 'total_departures']] = node_gdf[['total_arrivals', 'total_departures']].fillna(0)

# **Step 4: Scale Node Size Based on Arrivals and Departures**
min_size, max_size = 20, 200
node_gdf['size_arrivals'] = min_size + (node_gdf['total_arrivals'] / node_gdf['total_arrivals'].max()) * (max_size - min_size)
node_gdf['size_departures'] = min_size + (node_gdf['total_departures'] / node_gdf['total_departures'].max()) * (max_size - min_size)
#node_gdf[['size_arrivals', 'size_departures']] = node_gdf[['size_arrivals', 'size_departures']].fillna(min_size)
#power_factor = 1.5  # Adjust this to control scaling sensitivity
#node_gdf['size_arrivals'] = min_size + ((node_gdf['total_arrivals'] / node_gdf['total_arrivals'].max()) ** power_factor) * (max_size - min_size)
#node_gdf['size_departures'] = min_size + ((node_gdf['total_departures'] / node_gdf['total_departures'].max()) ** power_factor) * (max_size - min_size)



# **Step 5: Add Edges for Bristol Only**
edge_list = []
for idx, row in commuting_data.iterrows():
    if row['home_MSOA'] in G.nodes and row['work_MSOA'] in G.nodes and row['home_MSOA'] != row['work_MSOA']:
        G.add_edge(row['home_MSOA'], row['work_MSOA'], weight=row['commuter_count'])
        home_point = node_gdf.loc[node_gdf['MSOA_code'] == row['home_MSOA'], 'geometry'].values[0]
        work_point = node_gdf.loc[node_gdf['MSOA_code'] == row['work_MSOA'], 'geometry'].values[0]
        edge_list.append({'from': row['home_MSOA'], 'to': row['work_MSOA'], 'weight': row['commuter_count'], 'geometry': LineString([home_point, work_point])})

edge_gdf = gpd.GeoDataFrame(edge_list, crs="EPSG:4326")
edge_gdf = edge_gdf.to_crs(epsg=3857)

# **Ensure CRS is Correct for Mapping**
node_gdf = node_gdf.to_crs(epsg=3857)

# **Step 6: Plot Both Graphs**
fig, axes = plt.subplots(1, 2, figsize=(24, 12))

# **Plot Edges**
edge_gdf.plot(ax=axes[0], linewidth=edge_gdf['weight'] / edge_gdf['weight'].max() * 2, color="black", alpha=0.5)
edge_gdf.plot(ax=axes[1], linewidth=edge_gdf['weight'] / edge_gdf['weight'].max() * 2, color="black", alpha=0.5)

# **Plot Based on Arrivals**
node_gdf.plot(ax=axes[0], markersize=node_gdf['size_arrivals'], color="blue", alpha=0.8, edgecolor='black')
ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron)
axes[0].set_title("Bristol Commuting Network (Node Size by Arrivals)")
axes[0].set_axis_off()

# **Plot Based on Departures**
node_gdf.plot(ax=axes[1], markersize=node_gdf['size_departures'], color="red", alpha=0.8, edgecolor='black')
ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron)
axes[1].set_title("Bristol Commuting Network (Node Size by Departures)")
axes[1].set_axis_off()

plt.show()

