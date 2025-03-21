import json
import pandas as pd
from geopy.distance import geodesic

flows_df = pd.read_csv('commuter_flows.csv')
with open('Bris_Codes_with_Weights_and_Coords_NEW.json', 'r') as f:
    oa_coords = json.load(f)

origin_codes = set(flows_df['Home Cluster OA Code'])
dest_codes   = set(flows_df['Work Cluster OA Code'])
station_codes = origin_codes.union(dest_codes)
num_stations = len(station_codes)

station_locations = {code: tuple(oa_coords[code][1]) for code in station_codes if code in oa_coords}

connections = set()
for _, row in flows_df.iterrows():
    oa1 = row['Home Cluster OA Code']
    oa2 = row['Work Cluster OA Code']
    if oa1 == oa2:
        continue
    pair = tuple(sorted([oa1, oa2]))
    connections.add(pair)

total_track_length = 0.0
for oa1, oa2 in connections:
    coord1 = station_locations.get(oa1)
    coord2 = station_locations.get(oa2)
    if coord1 and coord2:
        distance_m = geodesic(coord1, coord2).meters
        total_track_length += distance_m

station_cost_total = num_stations * 700_000
track_cost_total   = total_track_length * 11_500
total_cost         = station_cost_total + track_cost_total

cost_per_station = total_cost / num_stations if num_stations > 0 else 0

print(f"Total building cost: £{total_cost:,.2f}")
print(f"Total track length: {total_track_length:,.2f} meters")
print(f"Cost per station: £{cost_per_station:,.2f}")
