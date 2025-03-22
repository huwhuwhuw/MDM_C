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
import numpy as np

"""
Do not share or abuse the api key. Save all results
Handles all generating routes with gmaps api, plotting graphs of clusters and routes between them and creating adjacency matrix
Use Data_Management class to easily open, save and handle all files needed
"""


def Plan_Route(start, end, api_key="""insert api key here"""):
    """


    Parameters
    ----------
    start : list
        OA code, latitude, longitude of start location.
    end : list
        OA code, latitude, longitude of end location.
    api_key : str, optional
        google maps project api key

    Returns
    -------
    None.

    """
    # initialise google maps library
    gmap = googlemaps.Client(key=api_key)

    # unpack start and end locations
    start_coords = [start[1], start[2]]
    end_coords = [end[1], end[2]]

    # get route between locations
    # gmaps can return multiple routes in a list, only ever want the first one
    route = gmap.directions(start_coords, end_coords, avoid=['highways', 'tolls'])[0]

    # save route to file
    filename = f"{start[0]}_{end[0]}"
    print(filename)
    with open(filename, 'wb+') as file:
        pkl.dump(route, file)

    print(f"{filename} route saved")


class Data_Management:
    """ Class for handling all file management relating to path finding
        data_folder : Name of folder to save and load data to
        csv_folder : Name of folder raw data files save and load to"""

    def __init__(self, data_folder='Road_Data', csv_folder='CSV_Files'):
        self.data_folder = os.path.join(os.path.dirname(__file__), data_folder)
        self.csv_folder = os.path.join(os.path.dirname(__file__), csv_folder)

    def save(self, data, filename):
        """ save any python object to the data folder """
        self.filepath = os.path.join(self.data_folder, filename)
        with open(self.filepath, 'wb') as file:
            pkl.dump(data, file)

    def load(self, filename):
        """ opens any python object from the data folder """
        self.filepath = os.path.join(self.data_folder, filename)
        with open(self.filepath, 'rb') as file:
            self.data = pkl.load(file)

        return self.data

    def load_coords(self, filename='msoa_lookup.csv'):
        """ loads coordinates of OA areas. I don't know why the file is named msoa it is the OA codes """
        self.coords = pd.read_csv(os.path.join(self.csv_folder, filename))

        return self.coords

    def load_OA_clusters(self, filename='commuter_flows.csv'):
        """ loads clusters from file """
        self.clusters = pd.read_csv(os.path.join(self.csv_folder, filename))
        # cluster locations are in both home and work columns (weirdly) and contain duplicates
        self.clusters = list(pd.concat([self.clusters['Home Cluster OA Code'], self.clusters['Work Cluster OA Code']]).unique())

        return self.clusters

    def load_cluster_coords(self, filename='msoa_lookup.csv'):
        """ loads coordinates for clusters """
        self.load_OA_clusters()
        self.coords = pd.read_csv(os.path.join(self.csv_folder, filename))
        self.coords = self.coords[self.coords['OA_code'].isin(self.clusters)]
        return self.coords

    def load_route(self, start, end):
        """ open and format gmaps route between start and end OA codes"""
        self.filepath = os.path.join(self.data_folder, f'{start}_{end}')
        with open(self.filepath, 'rb') as file:
            self.data = pkl.load(file)

        # gmaps returns a horrible html style result if you need more information make html copy and read manually
        self.route = {}
        self.route['distance'] = self.data['legs'][0]['distance']['value']
        self.route['steps'] = {}
        self.route['steps']['curves'] = [polyline.decode(curve['polyline']['points']) for curve in self.data['legs'][0]['steps']]

        # split steps into lat and long lists for pyplot
        self.route['steps']['lat'] = []
        self.route['steps']['long'] = []
        for curve in self.route['steps']['curves']:
            for point in curve:
                self.route['steps']['lat'].append(point[0])
                self.route['steps']['long'].append(point[1])

        return self.route

    def generate_graph(self, max_distance=4000):
        """ generate adjacency matrix of routes between clusters
            max_distance : int, maximum distance of routes between clusters in meters"""
        self.adj = pd.DataFrame(index=self.clusters, columns=self.clusters)  # adjaceny matrix for graph
        # iterate through every combination of
        for clusterA in self.clusters:
            for clusterB in self.clusters:
                try:
                    self.load_route(clusterA, clusterB)
                    if self.route['distance'] < max_distance:
                        self.adj.loc[clusterA, clusterB] = self.route['distance']
                except FileNotFoundError:
                    # some routes are only calculated one way, ignore attempts to load that route reversed
                    pass

        return self.adj


class OA_Plot:
    def __init__(self, handler):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        # import clustered OA codes and convert them to a geopandas dataframe
        self.clusters = handler.load_cluster_coords()
        self.clusters = {'geometry': [Point(long, lat) for lat, long in zip(self.clusters['Latitude'], self.clusters['Longitude'])]}
        self.clusters = gpd.GeoDataFrame(self.clusters, crs='EPSG:4326')

        # import google maps route
        self.curves = gpd.GeoDataFrame()

    def create_scatter_plot(self, **kwargs):
        """ function for making and saving a nice plot """
        self.clusters.plot(ax=self.ax, **kwargs)
        ctx.add_basemap(self.ax, crs=self.clusters.crs, source=ctx.providers.CartoDB.Positron)
        self.ax.set_title('Markov Clustered Locations')
        self.ax.grid()
        self.ax.set_xlabel('longitude')
        self.ax.set_ylabel('latitude')

    def create_route_plot(self, route):

        self.curves = pd.concat([self.curves, gpd.GeoDataFrame({'geometry': [Point(long, lat) for lat, long in zip(
            route['steps']['lat'], route['steps']['long'])]}, crs='EPSG:4326')])

        self.ax.plot(route['steps']['long'], route['steps']['lat'])

    def add_basemap(self):
        """ should be final call to ensure map generates properly """
        ctx.add_basemap(self.ax, crs=self.curves.crs, source=ctx.providers.CartoDB.Positron)

    def pretty_plot(self):
        self.ax.set_title('Route Map')
        self.ax.grid(alpha=0.4)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')


def Plot():

    handler = Data_Management()
    center = ('Center', 51.454919, -2.596510)

    handler.load_OA_clusters()
    figure = OA_Plot(handler)
    handler.clusters.append('center')
    adj = pd.DataFrame(index=handler.clusters, columns=handler.clusters)  # adjaceny matrix for graph analysis

    max_distance = 4000  # maximum distance to connect cluster by (meters)

    print(handler.generate_graph())
    return

    # mapping clusters together
    for clusterA in handler.clusters:
        for clusterB in handler.clusters:
            try:
                handler.load_route(clusterA, clusterB)
            except FileNotFoundError:
                # some routes are only calculated one way, ignore attempts to load that route reversed
                pass
            if handler.route['distance'] < max_distance:
                figure.create_route_plot(handler.route)
                adj.loc[clusterA, clusterB] = handler.route['distance']

    print(adj)

    figure.add_basemap()
    figure.pretty_plot()
    figure.fig.show()


def main():
    """ Create a complete graph of links between the clusters and an artificial center node """

    # my gmaps api key
    key = """ insert key here """

    # coordinates of colston street in bristol center where bus interchange currently is
    center = ('Center', 51.454919, -2.596510)

    # load cluster coordinates
    handler = Data_Management()
    clusters = handler.load_cluster_coords()

    # clusters.apply(lambda cluster: Plan_Route(center, cluster, api_key = key), axis=1)

    clusters.apply(lambda clusterA: clusters.apply(lambda clusterB: Plan_Route(clusterA, clusterB, api_key=key), axis=1), axis=1)

    # clusters.apply(center, Plan_Route, axis=1)


if __name__ == '__main__':
    # main()
    # Route_Cleaner('routeA')
    Plot()
    # Pickle_Test()
