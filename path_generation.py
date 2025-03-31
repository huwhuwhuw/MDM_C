import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
import networkx as nx
from scipy.spatial import KDTree

# from Indivual_Line_Score import Line_Score
import Indivual_Line_Score

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

    # check if route has already been calculated
    files = os.listdir(os.path.join(os.path.dirname(__file__), 'Road_Data'))
    filename = f"{start[0]}_{end[0]}"

    if filename not in files:
        # initialise google maps library
        gmap = googlemaps.Client(key=api_key)

        # unpack start and end locations
        start_coords = [start[1], start[2]]
        end_coords = [end[1], end[2]]

        # get route between locations
        # gmaps can return multiple routes in a list, only ever want the first one
        route = gmap.directions(start_coords, end_coords,
                                avoid=['highways', 'tolls'])[0]

        # save route to file
        with open(filename, 'wb+') as file:
            pkl.dump(route, file)
        print(f"{filename} route saved")

        return

    # don't call api and calculate route if file already exists
    elif filename in files:
        print(f'{filename} route already calculated')

        return


def generate_all_lines(clusterfile):
    """ Create a complete graph of links between the clusters and an artificial center node """

    key = 'AIzaSyCSNv7dxSrWgsP-0fAWGvecodPK2s29ycs'

    # load cluster coordinates
    handler = Data_Management()
    clusters = handler.load_cluster_coords(clusterfile)

    # clusters.apply(lambda cluster: Plan_Route(center, cluster, api_key = key), axis=1)

    clusters.apply(lambda clusterA: clusters.apply(
        lambda clusterB: Plan_Route(clusterA, clusterB, api_key=key), axis=1), axis=1)

    # clusters.apply(center, Plan_Route, axis=1)


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

    def load_OA_clusters(self, filename):
        """ loads clusters from file """
        # filename : name of csv file containing clusters
        # self.clusters = pd.read_csv(os.path.join(self.csv_folder, filename))
        # # cluster locations are in both home and work columns (weirdly) and contain duplicates
        # self.clusters = list(pd.concat(
        #     [self.clusters['Home Cluster OA Code'], self.clusters['Work Cluster OA Code']]).unique())

        self.clusters = pd.read_csv(os.path.join(self.csv_folder, filename))
        # self.clusters = list(self.clusters['OA_Code'])
        self.clusters = self.clusters['OA_Code'].unique()

        return self.clusters

    def load_cluster_coords(self, clusterfile, filename='msoa_lookup.csv'):
        """ loads coordinates for clusters """
        self.clusters = self.load_OA_clusters(clusterfile)
        self.coords = pd.read_csv(os.path.join(self.csv_folder, filename))
        self.coords = self.coords[self.coords['OA_Code'].isin(self.clusters)]
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
        self.route['steps']['curves'] = [polyline.decode(
            curve['polyline']['points']) for curve in self.data['legs'][0]['steps']]

        # split steps into lat and long lists for pyplot
        self.route['steps']['lat'] = []
        self.route['steps']['long'] = []
        for curve in self.route['steps']['curves']:
            for point in curve:
                self.route['steps']['lat'].append(point[0])
                self.route['steps']['long'].append(point[1])

        return self.route

    def dynamic_radius(self, k=5, alpha=3.5, filename='markov__with_count62.csv'):
        """ implements a dynamic radius to generate graph from
            nodes with closer neighbours have a lower maximum connection radius """

        # create copy of coordinates to convert to lat/long without impacting other functions
        coordinates = self.load_cluster_coords(clusterfile=filename)

        # radius has to be stored with coords to allow oa code indexing
        coordinates['radius'] = np.nan

        # pretend the earth isnt round and convert lat/long to meters
        coordinates['Latitude'] = coordinates['Latitude'].map(
            lambda x: x*111320)
        coordinates['Longitude'] = coordinates['Longitude'].map(
            lambda x: 40075000*math.cos(x)/360)

        coordinate_list = [[long, lat] for long, lat in zip(
            self.coords['Longitude'], self.coords['Latitude'])]
        self.tree = KDTree(coordinate_list)

        for cluster in self.clusters:
            # get distance of k nearest nodes from every node
            self.distance, _ = self.tree.query([self.coords.loc[self.coords['OA_Code'] == cluster, 'Longitude'].iloc[0],
                                                self.coords.loc[self.coords['OA_Code'] == cluster, 'Latitude'].iloc[0]],
                                               k=k)

            coordinates.loc[coordinates['OA_Code'] == cluster,
                            'radius'] = np.mean(self.distance) * alpha

        return coordinates

    def generate_graph(self, clusterfile, k=5, alpha=4):
        """ generate adjacency matrix of routes between clusters
            max_distance : int, maximum distance of routes between clusters in meters"""

        self.load_OA_clusters(clusterfile)

        # create maximum radius for each node
        coords = self.dynamic_radius(k, alpha, filename=clusterfile)

        # adjaceny matrix for graph
        self.adj = pd.DataFrame(index=self.clusters, columns=self.clusters)
        # iterate through every combination of clusters
        for clusterA in self.clusters:
            for clusterB in self.clusters:
                try:
                    self.load_route(clusterA, clusterB)
                    if self.route['distance'] < coords.loc[coords['OA_Code'] == clusterA, 'radius'].iloc[0]:
                        self.adj.loc[clusterA,
                                     clusterB] = self.route['distance']
                except FileNotFoundError:
                    # some routes are only calculated one way, ignore attempts to load that route reversed
                    print(f'Route between {clusterA} and {clusterB} not found')

        return self.adj


class OA_Plot:
    def __init__(self, handler, clusterfile):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        # import clustered OA codes and convert them to a geopandas dataframe
        self.clusters = handler.load_cluster_coords(clusterfile)
        self.clusters = {'geometry': [Point(long, lat) for lat, long in zip(
            self.clusters['Latitude'], self.clusters['Longitude'])]}
        self.clusters = gpd.GeoDataFrame(self.clusters, crs='EPSG:4326')

        # import google maps route
        self.curves = gpd.GeoDataFrame()

    def create_scatter_plot(self, **kwargs):
        """ function for making and saving a nice plot """
        self.clusters.plot(ax=self.ax, **kwargs)
        # self.ax.set_title('Markov Clustered Locations')

    def create_route_plot(self, route, **kwargs):

        self.curves = pd.concat([self.curves, gpd.GeoDataFrame({'geometry': [Point(long, lat) for lat, long in zip(
            route['steps']['lat'], route['steps']['long'])]}, crs='EPSG:4326')])

        self.ax.plot(route['steps']['long'], route['steps']['lat'], **kwargs)

    def add_basemap(self):
        """ should be final call to ensure map generates properly """
        ctx.add_basemap(self.ax, crs=self.clusters.crs,
                        source=ctx.providers.CartoDB.Positron)

    def pretty_plot(self):
        # self.ax.set_title('Route Map')
        self.ax.grid(alpha=0.4)
        # self.ax.set_xlabel('Longitude')
        # self.ax.set_ylabel('Latitude')


def Plot(k, alpha, clusterfile='markov__with_count62.csv'):
    """ plots all clusters and route for current cluster method """

    # load clusters and initialise plot
    handler = Data_Management()
    handler.load_OA_clusters(clusterfile)
    radius_coords = handler.dynamic_radius(k, alpha, filename=clusterfile)
    figure = OA_Plot(handler, clusterfile)

    max_distance = 2000  # maximum distance to connect cluster by (meters)

    # mapping routes between clusters
    for clusterA in handler.clusters:
        for clusterB in handler.clusters:
            try:
                handler.load_route(clusterA, clusterB)
                if handler.route['distance'] < radius_coords.loc[handler.coords['OA_Code'] == clusterA, 'radius'].iloc[0]:
                    figure.create_route_plot(handler.route, color='cadetblue')

            except FileNotFoundError:
                # some routes are only calculated one way, ignore attempts to load that route reversed
                # or if using a different cluster set some routes may no be calculated
                print(f'Route between {clusterA} and {clusterB} not found')

    # add pyplot parameters like color, size, etc to create_scatter_plot as usual
    figure.create_scatter_plot(color='black')
    figure.add_basemap()
    figure.pretty_plot()
    figure.ax.set_title(f'k={k}, alpha={alpha}')
    figure.fig.show()

    return figure


def create_routes(k, alpha, count, center, clusterfile, plot=False):
    """ create optimal routes for the given clusters """
    # generate the directional graph

    handler = Data_Management()
    adjacency_matrix = handler.generate_graph(clusterfile, k=k, alpha=alpha)
    adjacency_matrix.fillna(0, inplace=True)
    network = nx.from_pandas_adjacency(
        adjacency_matrix, create_using=nx.DiGraph)

    # calculate most central betweenness node
    between_center = nx.betweenness_centrality(network)

    handler.load_cluster_coords(clusterfile)
    between = handler.coords.loc[handler.coords['OA_Code'] == max(
        between_center, key=between_center.get)]

    routes = []

    # iteratively create routes removing already visited nodes each time
    while len(routes) < count:

        potential_routes = []
        for terminal in handler.clusters:
            if terminal == center:
                # don't calculate route from center to itself
                pass
            else:
                try:
                    potential_routes.append(
                        nx.shortest_path(network, center, terminal))
                except:
                    # if there is no path between nodes skip the node
                    print(f'No path between {center} and {terminal} found')

        coords = import_json()

        line_scores = []

        for route in potential_routes:
            line_scores.append(Indivual_Line_Score.Line_Score(route, coords))

        try:
            # add the route nodes and total score to the record
            routes.append(
                [potential_routes[line_scores.index(max(line_scores))], max(line_scores)])
        except:
            # if no routes can be found return
            print(f'No routes found')
            break

        # remove the visited route from the graph while keeping the center
        # print(routes)
        nodes_to_drop = routes[-1][0].copy()
        # nodes_to_drop.remove(center)
        # print(nodes_to_drop[-1])
        try:
            network.remove_nodes_from(nodes_to_drop[-3:])
        except:
            print(f'Route length too short')
            break

    if plot:
        figure = OA_Plot(handler, clusterfile)
        route_plot(routes, figure)

    return routes


def route_plot(routes, figure):

    # plotting only the best routes
    handler = Data_Management()
    colors = ['red', 'green', 'blue', 'gold', 'purple', 'black']
    patches = []

    for i, route in enumerate(routes):
        for cluster_index in range(0, len(route[0])-1):
            # plot each route segment seperately
            segment = handler.load_route(
                route[0][cluster_index], route[0][cluster_index+1])

            figure.create_route_plot(
                segment, color=colors[i])
        patches.append(mpatches.Patch(color=colors[i], label=f'Route {i+1}'))

    figure.ax.legend(handles=patches)
    figure.create_scatter_plot()
    figure.add_basemap()
    figure.ax.set_xticks([])
    figure.ax.set_yticks([])

    print(f'highest scoring lines: {routes}')


def import_json():
    # filepath = 'C:/Users/pigwi/Coding/MDM3/Transport/Bris_Codes_with_Weights_and_Coords_NEW.json'
    filepath = 'C:/Users/Kate/MDM_C/Bris_Codes_with_Weights_and_Coords_NEW.json'
    with open(filepath, 'r') as file:
        import json
        Dict = json.load(file)
    return Dict


def hyperparameter_optimisation(k, alpha, center, clusterfile, count=1):
    """ Handles creating and plotting route scores for graph generation hyperparameter optimisation """
    """ k : int or array of k to test over
        alpha : int or array of alpha to test over
        clusterfile : str of clusters to create route for
        count : int number of lines to create """

    handler = Data_Management()
    fig, ax = plt.subplots()
    routes = []
    route_scores = []

    for k_curr in k:
        for alpha_curr in alpha:
            route = create_routes(k_curr, alpha_curr, count, center, clusterfile)

            # attempt to save the current route
            try:
                # if a route is found add it to the saved route scores
                route_scores.append({'k': k_curr,
                                    'alpha': alpha_curr,
                                     'score': route[0][1]})
                # store route to plot later if a route is found
                routes.append(route)
            except:
                print(f'No route found for k={k_curr}, alpha={alpha_curr}')

        print(f'k={k_curr} complete')

    print(route_scores)
    max_score = max([route['score'] for route in route_scores])

    # plot heatmap of line scores for k, alpha combinations, converting to array of pixel colours for imshow
    grid = np.zeros((len(alpha), len(k)))
    for route in route_scores:
        grid[route['alpha']-1, route['k']-1] = route['score']

    # flip along y axis so (0, 0) is in bottom left
    grid = np.flip(grid, 0)

    ax.imshow(grid)
    ax.set_xticks(list(map(lambda x: x-1, k)), k)
    ax.set_yticks(list(map(lambda x: x-1, alpha)), list(reversed(alpha)))

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\alpha')

    fig.tight_layout()
    fig.savefig('kmed_grid.pdf', format='pdf')


def route_length(route):
    handler = Data_Management()
    distance = 0
    for index in range(0, len(route)-1):
        distance += handler.load_route(route[index], route[index+1])['distance']

    print(distance)


def top_3_routes():
    # k=6, alpha=3 markov cluster top 3 routes
    routeMarkov = [['E00174312', 'E00073425', 'E00073526', 'E00073500', 'E00073545', 'E00073268', 'E00074057', 'E00073856', 'E00075639', 'E00073385', 'E00073659', 'E00073325'],
                   ['E00174312', 'E00073425', 'E00073526', 'E00073500', 'E00073545',
                       'E00073268', 'E00074057', 'E00073856', 'E00075639', 'E00075602'],
                   ['E00174312', 'E00073425', 'E00073526', 'E00073500', 'E00073545', 'E00073268', 'E00074025', 'E00074000', 'E00073394', 'E00073380']]
    # k=5, alpha=3 k-medoids clusters top 3 routes
    routeKmed = [['E00174218', 'E00174289', 'E00073543', 'E00073283', 'E00073562', 'E00074002', 'E00074428', 'E00073234', 'E00073336'],
                 ['E00174218', 'E00174289', 'E00073543', 'E00073283', 'E00073562', 'E00174316', 'E00073436', 'E00174245'],
                 ['E00174218', 'E00174289', 'E00073543', 'E00073283', 'E00073562', 'E00074120', 'E00073868']]

    clusterMarkov = 'markov_with_count62.csv'
    clusterKmed = 'kmedoids_with_count50.csv'

    handler = Data_Management()
    markov = OA_Plot(handler, clusterMarkov)
    kmed = OA_Plot(handler, clusterKmed)

    colors = ['red', 'green', 'blue']
    patches = []

    # markov plot
    for k, route in enumerate(routeMarkov):
        patches.append(mpatches.Patch(color=colors[k], label=f'Route {k+1}'))
        for i, node in enumerate(route):
            try:
                handler.load_route(route[i], route[i+1])
                markov.create_route_plot(handler.route, color=colors[k])

                if i == 0:
                    markov.ax.scatter(handler.route['steps']['long'][0], handler.route['steps']['lat'][0])
            except:
                pass

    markov.create_scatter_plot(alpha=0)
    markov.ax.legend(handles=patches)
    markov.ax.set_xticks([])
    markov.ax.set_yticks([])
    markov.ax.set_title('Top 3 Routes for Markov Clusters')
    markov.add_basemap()

    # kmed plot
    for k, route in enumerate(routeKmed):
        # patches.append(mpatches.Patch(color=colors[k], label=f'Route {k+1}'))
        for i, node in enumerate(route):
            try:
                handler.load_route(route[i], route[i+1])
                kmed.create_route_plot(handler.route, color=colors[k])

                if i == 0:
                    kmed.ax.scatter(handler.route['steps']['long'][0], handler.route['steps']['lat'][0])
            except:
                pass

    kmed.create_scatter_plot(alpha=0)
    kmed.ax.legend(handles=patches)
    kmed.ax.set_xticks([])
    kmed.ax.set_yticks([])
    kmed.ax.set_title('Top 3 Routes for K-medoids Clusters')
    kmed.add_basemap()


def main():
    pass


if __name__ == '__main__':
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    count = 3   # maximum 6

    main()

    # filename of clusters; 40, 50, 60, markov
    clusterfile = 'kmedoids_with_count50.csv'
    center = 'E00174218'    # k-medoids 50-cluster most travelled cluster (stoke croft)
    # clusterfile = 'markov_with_count62.csv'
    # center = 'E00174312'  # markov 62-cluster most travelled cluster (hospital in center)

    # generate_all_lines(clusterfile)
    # hyperparameter_optimisation(k, alpha, center, clusterfile)

    k = 5
    alpha = 3

    # figure = Plot(k, alpha, clusterfile)     # plots full graph of network
    # handler = Data_Management()
    # figure = OA_Plot(handler, clusterfile)
    # routes = create_routes(k, alpha, count, center, clusterfile, plot=False)  # create routes for given parameters
    # route_plot(routes, figure)
    # figure.ax.set_title('Top 3 Routes, K-medoids Clusters')

    # Route_Cleaner('routeA')
    # Pickle_Test()
