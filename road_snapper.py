import matplotlib.pyplot as plt
import geopandas as gpd
import os
import math
import googlemaps
import pickle as pkl
import pandas as pd
import contextily as ctx
from shapely.geometry import Point
import polyline

"""
Do not call the api repeatidly it costs moiney. Save all results with pickle
"""


class Data_Management:
    """ Class for handling all file management relating to path finding """
    def __init__(self, data_folder='Road_Data', csv_folder = 'CSV_Files'):
        self.data_folder = os.path.join(os.path.dirname(__file__), data_folder)
        self.csv_folder = os.path.join(os.path.dirname(__file__), csv_folder)

    def save(self, data, filename):
        self.filepath = os.path.join(self.data_folder, filename)
        with open(self.filepath, 'wb') as file:
            pkl.dump(data, file)

    def load(self, filename):
        self.filepath = os.path.join(self.data_folder, filename)
        with open(self.filepath, 'rb') as file:
            data = pkl.load(file)
        return data
    
    def load_coords(self, filename='msoa_lookup.csv'):
        return pd.read_csv(os.path.join(self.csv_folder, filename))
    
    def load_OA_clusters(self, filename='commuter_flows.csv'):
        clusters = pd.read_csv(os.path.join(self.csv_folder, filename))
        return clusters['Home Cluster OA Code'].unique()
    
    def load_cluster_coords(self):
        self.coords = self.load_coords()
        self.clusters = self.load_OA_clusters()
        
        return self.coords[self.coords['OA_code'].isin(self.clusters)]


class OA_Plot:
    def __init__(self):
        self.fig = plt.subplot()
        self.nodes = gpd.GeoDataFrame({'geometry': [None]})
        

    def add_node(self, node, **kwargs):
        """ Code must contain latitudeand longitude coordinates """
        self.nodes.append(node)
        node.plot(ax=self.fig)
    
    def add_basemap(self):
        ctx.add_basemap(self.fig, source=ctx.providers.CartoDB.Positron)



def Pickle_Test():
    """ function for testing the Data_Management class """
    # test = [x for x in range(0, 10)]
    data = Data_Management()
    # data.save(test, 'test')

    print(data.load_coords().info())


def Plot():

    # initialise file manager
    handler = Data_Management()

    coords = handler.load_cluster_coords()

    gdf = gpd.GeoDataFrame()

    geometry = {'geometry': [Point(long, lat) for lat, long in zip(coords['Latitude'], coords['Longitude'])]}

    # simple plot of bristol and 
    data = gpd.GeoDataFrame(geometry, crs='EPSG:4326').to_crs(epsg=3857)
    ax = data.plot(figsize=(12, 9))
    ctx.add_basemap(ax, crs=data.crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
    print(data.crs)
    print(data.total_bounds)

def Route_Cleaner(route_path):
    
    # gmaps uses a horrible json format for the steps so good luck decoding it
    handler = Data_Management()
    route = handler.load(route_path)[0]['legs'][0]['steps']
    

    # pull out start, end, and encoded curves for each step
    curves_encoded = [step['polyline']['points'] for step in route]
    latitude = [step['start_location']['lat'] for step in route]
    longitude = [step['end_location']['lng'] for step in route]
    
    steps = [[step['start_location']['lat'], step['end_location']] for step in route]
    
    curves_decoded = [polyline.decode(curve) for curve in curves_encoded]
    
    lat = []
    long = []
    for x in curves_decoded:
        for y in x:
            lat.append(y[0])
            long.append(y[1])
    

    route_gdf = gpd.GeoDataFrame({'geometry': [Point(long, lat) for lat, long in zip(latitude, longitude)]},
                                 crs='EPSG:4326')

    

    
    ax = route_gdf.plot(figsize=(12, 9))
    ax.plot(long, lat)
    ctx.add_basemap(ax, crs=route_gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)



def main():
    # google maps api key
    api_key = None

    # initialise google maps library
    gmap = googlemaps.Client(key=api_key)

    # initialise data management class
    handler = Data_Management()

    bristol_center = (51.4545, -2.5879)  # Latitude, Longitude

    # import clusters and grab first and last as they are far apart
    clusters = handler.load_cluster_coords()
    start = clusters.iloc[0]
    end = clusters.iloc[-1]

    start_coords = [start['Latitude'], start['Longitude']]
    end_coords = [end['Latitude'], end['Longitude']]
    
    print(start_coords)
    print(end_coords)
    
    # get route between locations
    # route = gmap.directions(start_coords, end_coords)

    # handler.save(route, 'routeA')
    # print(route)


    
    

if __name__ == '__main__':
    # main()
    Route_Cleaner('routeA')
    # Plot()
    # Pickle_Test()