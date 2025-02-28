#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:13:53 2025

@author: huwtebbutt
"""
import csv
import requests
file_path = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Output_area_(2021)_to_future_Parliamentary_Constituencies_Lookup_in_England_and_Wales.csv'

Bristol_Codes=[]
with open(file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if 'Bristol, City of' in row:
            print(row[0])
            area_code = f'areas/{row[0]}'
            url = f'https://findthatpostcode.uk/{area_code}.json'
            Example_Postcodes=[]
            response = requests.get(url)
            if response.status_code == 200:
                D = response.json()
                Data=D['data']
                Relationships=Data['relationships']
                Postcodes=Relationships['example_postcodes']
                Postcode_data=Postcodes['data']
                for post in Postcode_data:
                    Ex_Post=post['id'].replace(" ", "+")
                    Example_Postcodes.append(Ex_Post)
            else:
                print(f"Error: {response.status_code}")


            Lats=[]
            Longs=[]
            for P in Example_Postcodes:
                postcode=f'postcodes/{P}'
                url = f'https://findthatpostcode.uk/{postcode}.json'
                response = requests.get(url)
                if response.status_code == 200:
                    D = response.json()
                    data=D['data']
                    attr=data['attributes']
                    lattitude=attr['lat']
                    longitude=attr['long']
                    Lats.append(lattitude)
                    Longs.append(longitude)
                else:
                    print(f"Error: {response.status_code}")
            Lat=sum(Lats)/len(Lats)
            Long=sum(Longs)/len(Longs)
            new_row=[row[0],Lat,Long]
            Bristol_Codes.append(new_row)

filename = '/Users/huwtebbutt/Documents/MDM3/3rd_term/Bristol_Codes_With_Coords.csv'

# Writing to the CSV file
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(Bristol_Codes)




