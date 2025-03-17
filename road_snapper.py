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
"""


def Plan_Router(api_key="""insert api key here"""):

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
    route = gmap.directions(start_coords, end_coords)

    # overwrites file currently there
    handler.save(route, 'routeA')
    print(route)


class Data_Management:
    """ Class for handling all file management relating to path finding
        data_folder : Name of folder to save and load data to
        csv_folder : Name of folder raw data files save and load to"""

    def __init__(self, data_folder='Road_Data', csv_folder='CSV_Files'):
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
    def __init__(self, handler):
        self.fig, self.ax = plt.subplots(figsize=(12, 9))

        # import clustered OA codes and convert them to a geopandas dataframe
        self.clusters = handler.load_cluster_coords()
        self.clusters = {'geometry': [Point(long, lat) for lat, long in zip(self.clusters['Latitude'], self.clusters['Longitude'])]}
        self.clusters = gpd.GeoDataFrame(self.clusters, crs='EPSG:4326')

        # import google maps route
        self.route = handler.load('routeA')[0]['legs'][0]['steps']

    def create_scatter_plot(self, **kwargs):
        """ function for making and saving a nice plot """
        self.clusters.plot(ax=self.ax, **kwargs)
        ctx.add_basemap(self.ax, crs=self.clusters.crs, source=ctx.providers.CartoDB.Positron)
        self.ax.set_title('Markov Clustered Locations')
        self.ax.grid()
        self.ax.set_xlabel('longitude')
        self.ax.set_ylabel('latitude')
        # self.ax.set_yticks(np.linspace(6716000, 6702000, num=8))
        # self.ax.set_yticklabels(np.linspace(6716000, 6702000, num=8))

    def create_route_plot(self, **kwargs):
        # pull out start, end, and encoded curves for each step
        self.curves_encoded = [step['polyline']['points'] for step in self.route]
        self.curves_decoded = [polyline.decode(curve) for curve in self.curves_encoded]

        self.curve_points = {'lat': [],
                             'long': []
                             }

        for x in self.curves_decoded:
            for y in x:
                self.curve_points['lat'].append(y[0])
                self.curve_points['long'].append(y[1])

        self.curves = gpd.GeoDataFrame({'geometry': [Point(long, lat) for lat, long in zip(
            self.curve_points['lat'], self.curve_points['long'])]}, crs='EPSG:4326')

        self.ax.plot(self.curve_points['long'], self.curve_points['lat'])
        ctx.add_basemap(self.ax, crs=self.curves.crs, source=ctx.providers.CartoDB.Positron)

        self.ax.set_title('Planned Route')
        self.ax.grid(alpha=0.4)
        self.ax.set_xlabel('longitude')
        self.ax.set_ylabel('latitude')

    def save_plot(self, filename):
        self.fig.savefig()


def Plot_Clusters():
    handler = Data_Management()
    bristol = OA_Plot(handler)
    bristol.create_scatter_plot()


def main():
    handler = Data_Management()
    figure = OA_Plot(handler)
    figure.create_route_plot()
    figure.create_scatter_plot()


if __name__ == '__main__':
    main()
    # Route_Cleaner('routeA')
    # Plot()
    # Pickle_Test()
