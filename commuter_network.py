import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import contextily as ctx
from geopy.distance import geodesic

# Load MSOA lookup CSV
msoa_lookup = pd.read_csv(
    r"C:\Users\kitcr\Downloads\MDM3 Trams\Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V3_-8386444323138516297.csv")
msoa_lookup.rename(columns={'MSOA21CD': 'MSOA_code', 'LAT': 'Latitude', 'LONG': 'Longitude'}, inplace=True)

# Load commuting data
commuting_data = pd.read_csv(r"C:\Users\kitcr\Downloads\MDM3 Trams\bristol_MSOA_data.csv")
commuting_data.rename(columns={
    'Middle layer Super Output Areas code': 'home_MSOA',
    'MSOA of workplace code': 'work_MSOA',
    'Count': 'commuter_count'
}, inplace=True)

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

# **Filter Nodes: Only Include Those Within Bristol**
bristol_nodes = msoa_lookup[msoa_lookup.apply(
    lambda row: is_within_radius(row['Latitude'], row['Longitude'], bristol_center, radius_km), axis=1)]

# **Step 1: Create the Directed Graph**
G = nx.DiGraph()

# Add only Bristol nodes
for idx, row in bristol_nodes.iterrows():
    G.add_node(row['MSOA_code'], pos=(row['Longitude'], row['Latitude']))

# **Step 2: Compute Node Size Based Only on `work_MSOA` Appearances (Within Bristol)**
bristol_commuting_data = commuting_data[
    (commuting_data['home_MSOA'].isin(bristol_nodes['MSOA_code'])) &
    (commuting_data['work_MSOA'].isin(bristol_nodes['MSOA_code']))  # Ensure work nodes are also within Bristol
]

node_sizes = bristol_commuting_data.groupby('work_MSOA')['commuter_count'].sum().rename("total_arrivals")

# **Step 3: Merge Arrival Data into Node GeoDataFrame**
node_positions = nx.get_node_attributes(G, 'pos')
node_gdf = gpd.GeoDataFrame(
    {'MSOA_code': list(node_positions.keys())},
    geometry=gpd.points_from_xy([pos[0] for pos in node_positions.values()],
                                [pos[1] for pos in node_positions.values()]),
    crs="EPSG:4326"
)

# Merge arrival data into the node dataframe
node_gdf = node_gdf.merge(node_sizes, left_on='MSOA_code', right_index=True, how='left')
node_gdf['total_arrivals'] = node_gdf['total_arrivals'].fillna(0)

# **Step 4: Scale Node Size Based on Total Arrivals (Only from `work_MSOA`)**
min_size, max_size = 20, 200
node_gdf['size'] = min_size + (node_gdf['total_arrivals'] / node_gdf['total_arrivals'].max()) * (max_size - min_size)
node_gdf['size'] = node_gdf['size'].fillna(min_size)  # Ensure no NaN values

# **Step 5: Add Edges for Bristol Only**
for idx, row in bristol_commuting_data.iterrows():
    if row['home_MSOA'] in G.nodes and row['work_MSOA'] in G.nodes and row['home_MSOA'] != row['work_MSOA']:
        G.add_edge(row['home_MSOA'], row['work_MSOA'], weight=row['commuter_count'])

# Convert edges to GeoDataFrame
edge_list = []
for u, v, d in G.edges(data=True):
    home_node = node_gdf.loc[node_gdf['MSOA_code'] == u, 'geometry']
    work_node = node_gdf.loc[node_gdf['MSOA_code'] == v, 'geometry']

    if not home_node.empty and not work_node.empty:
        home_point = home_node.iloc[0]
        work_point = work_node.iloc[0]
        edge_list.append({
            'from': u,
            'to': v,
            'weight': d['weight'],
            'geometry': LineString([home_point, work_point])
        })

edge_gdf = gpd.GeoDataFrame(edge_list, crs="EPSG:4326")

# **Ensure CRS is Correct for Mapping**
node_gdf = node_gdf.to_crs(epsg=3857)
edge_gdf = edge_gdf.to_crs(epsg=3857)

# **Step 6: Plot the Fixed Graph with Arrows & Thickness**
fig, ax = plt.subplots(figsize=(12, 12))

# Plot edges with varying thickness
for idx, row in edge_gdf.iterrows():
    x, y = row['geometry'].xy
    line_width = (row['weight'] / edge_gdf['weight'].max()) * 0.5  # Scale thickness

    ax.plot(x, y, linewidth=line_width, alpha=0.7, color="black")

    # Compute midpoint for the arrow
    mid_x = (x[0] + x[1]) / 2
    mid_y = (y[0] + y[1]) / 2

    # Add arrows for direction with thickness matching line width
    ax.annotate("",
                xy=(mid_x, mid_y), xycoords='data',
                xytext=(x[0], y[0]), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="black", lw=line_width))

# Plot nodes with size based on arrivals
node_gdf.plot(ax=ax, markersize=node_gdf['size'], color="blue", alpha=0.8, edgecolor='black')

# **Fix basemap projection**
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# **Legend**
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color='blue', label='Nodes sized by commuter arrivals (Bristol Only)')]
ax.legend(handles=legend_patches, loc='upper right')

# **Finalize plot**
ax.set_title("Bristol Commuting Network (Scaled Node Size by Arrivals)")
ax.set_axis_off()

plt.show()
