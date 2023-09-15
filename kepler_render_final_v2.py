from flask import Flask
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from keplergl import KeplerGl
import requests
import geopandas as gpd
from shapely.geometry import Point, Polygon
import networkx as nx
import osmnx as ox
ox.config(log_console=True, use_cache=True)

######################### analyze the base facility data and calculate Z-score ##############################################

# Load the CSV data into a DataFrame
df = pd.read_csv('NCR GenxFacilityData.csv', sep='|')

# Function to categorize facilities based on z-scores and additional conditions
def categorize_utilization_zscore(row):
    # Check if the facility is non-operational based on 'Operational Status'
    if row['Operational Status'] == 'Non-Operational' or row['opeational_days_in_month'] == 0:
        return 0.0
    
    # Calculate the z-score for the 'Percentage_utilization' column
    mean_utilization = df['Percentage_utilization'].mean()
    std_deviation = df['Percentage_utilization'].std()
    z_score = (row['Percentage_utilization'] - mean_utilization) / std_deviation
    
    # Check the additional conditions
    if pd.isna(row['Total_test_completed']) or row['Total_test_completed'] == 0:
        return 0.0
    else:
        return z_score

# Function to categorize facilities based on z-scores and additional conditions
def categorize_utilization_zscore_with_conditions(row):
    # Check if the facility is non-operational based on 'Operational Status'
    if row['Operational Status'] == 'Non-Operational' or row['opeational_days_in_month'] == 0:
        return 'Not Operational'
    
    # Calculate the z-score for the 'Percentage_utilization' column
    mean_utilization = df['Percentage_utilization'].mean()
    std_deviation = df['Percentage_utilization'].std()
    z_score = (row['Percentage_utilization'] - mean_utilization) / std_deviation
    
    # Check the additional conditions
    if pd.isna(row['Total_test_completed']) or row['Total_test_completed'] == 0:
        return 'Underutilized'
    elif z_score < -1:
        return 'Underutilized'
    elif -1 <= z_score <= 1:
        return 'Optimally Utilized'
    else:
        return 'Overutilized'

# Apply categorization function to each row

df['mean_utilization'] = df['Percentage_utilization'].mean()
df['std_deviation']  = df['Percentage_utilization'].std()

df['zscore_calculated'] = df.apply(categorize_utilization_zscore, axis=1)
    
# Apply categorization function to each row
df['UtilizationCategory'] = df.apply(categorize_utilization_zscore_with_conditions, axis=1)

# Add source coordinates (actual lat/lon values)
source_latitude = 14.5786085
source_longitude = 120.9861

# Add source coordinates (radian-converted)
df['Source_lat'] = source_latitude
df['Source_lon'] = source_longitude
df['Source_lat_rad'] = np.radians(source_latitude)
df['Source_lon_rad'] = np.radians(source_longitude)

# Convert facility latitude and longitude from degrees to radians
df['Latitude_rad'] = np.radians(df['Latitude'])
df['Longitude_rad'] = np.radians(df['Longitude'])

# Calculate Haversine distance between source and facilities (convert to km)
def haversine_distance(row):
    source = np.array([row['Source_lat_rad'], row['Source_lon_rad']]).reshape(1, -1)
    facility = np.array([row['Latitude_rad'], row['Longitude_rad']]).reshape(1, -1)
    distances = haversine_distances(source, facility) * 6371  # Convert to km
    return distances[0][0]

df['Distance_to_Source_km'] = df.apply(haversine_distance, axis=1)

# Define clustering radii
cluster_radii = [2, 5, 10]  # 2km, 5km, and 10km

# Apply custom clustering based on distances
def custom_clustering(distance_km):
    if distance_km < 5:
#         return 'Cluster_2km'
#     elif 2 <= distance_km < 5:
        return 'Cluster_5km'
    elif 5 <= distance_km < 10:
        return 'Cluster_10km'
    else:
        return 'Cluster_gthn_10km'

# # Choose the number of clusters 
# n_clusters = 3

# # Create the K-Means model
# kmeans = KMeans(n_clusters=n_clusters, random_state=42) -- check the random_state variable value

# # Fit the model to the data
# kmeans.fit(X) 

# # Add cluster labels to the DataFrame
# df['Cluster'] = kmeans.labels_

# # Define cluster names based on your interpretation of the clusters
# cluster_names = {
#     0: 'Underutilized',
#     1: 'Optimally Utilized',
#     2: 'Overutilized'
# }

# # Map cluster labels to cluster names
# df['UtilizationCategory'] = df['Cluster'].map(cluster_names)
    
df['Cluster'] = df['Distance_to_Source_km'].apply(custom_clustering)
df['source_facility_icon'] = 'car'
df['target_facility_icon'] = 'place'

# Save the categorized and clustered data back to a CSV file
df.to_csv('categorized_clustered_data.csv', index=False, sep='|')  # Replace with your desired output file path


################# calculcate the driving distance using mapquest API later on this will be replaced with google API #############################

#/************************************* google API code ******************************************
# # Function to calculate driving route and return route details
# def calculate_driving_route_google_api(row):
#     # Define source and destination coordinates
#     source_coords = (row['Source_lat'], row['Source_lon'])
#     dest_coords = (row['Latitude'], row['Longitude'])

#     # Request driving directions from Google Maps API
#     directions_result = gmaps.directions(
#         source_coords, dest_coords,
#         mode="driving",
#         departure_time=datetime.now(),
#     )

#     if directions_result:
#         # Extract relevant information from the API response
#         route = directions_result[0]['legs'][0]
#         driving_distance = route['distance']['text']
#         driving_time = route['duration']['text']
#         polyline = route['overview_polyline']['points']

#         return {
#             'DrivingDistance': driving_distance,
#             'DrivingTime': driving_time,
#             'Polyline': polyline
#         }
#     else:
#         return None

# # Apply the function to calculate driving route for each row -- using google API
# route_info = df.apply(calculate_driving_route_google_api, axis=1)

# # Add the route information to the DataFrame
# df = pd.concat([df, route_info], axis=1)
# ************************************************************************************************/

# Function to calculate driving route and return route details
def calculate_driving_route(row, api_key):
   # Define source and destination coordinates
   source_coords = f"{row['stRider_location_lat']},{row['stRider_location_long']}"
   dest_coords = f"{row['Latitude']},{row['Longitude']}"

   # Construct the request URL for MapQuest Directions API
   base_url = "https://www.mapquestapi.com/directions/v2/route"
   params = {
       "key": api_key,
       "from": source_coords,
       "to": dest_coords,
       "routeType": "fastest",
       "doReverseGeocode": "false",
       "unit": "k",
       "locale": "en_US"
   }

   # Make the API request
   response = requests.get(base_url, params=params)

   if response.status_code == 200:
       data = response.json()
       if data["info"]["statuscode"] == 0:
           route = data["route"]
           driving_distance = route["distance"]
           driving_time = route["formattedTime"]
        # this will give you complete json of route with step wise directions            
#            polyline = route["legs"][0]["maneuvers"] 
        # code to just the get the line string of the actual path is as below 
        # Extract polyline points from maneuvers
           maneuvers = route["legs"][0]["maneuvers"]
           polyline_points = [(maneuver["startPoint"]["lng"], maneuver["startPoint"]["lat"]) for maneuver in maneuvers]
           # Format the route as LINESTRING
#            linestring = "LINESTRING (" + ", ".join([f"{lon} {lat}" for lat, lon in zip(polyline_points[1::2], polyline_points[::2])]) + ")"
           linestring = "LINESTRING (" + ", ".join([f"{lon} {lat}" for lat, lon in zip(polyline_points[1::2], polyline_points[::2])]) + ")"
           linestring = linestring.replace("((", "(").replace("))", ")").replace(", "," ").replace(") (",",")
           return {
               'DrivingDistance': driving_distance,
               'DrivingTime': driving_time,
#                'Polyline': polyline
               'route_path_string': linestring    
           }
       else:
           return None
   else:
       return None

# Replace 'YOUR_MAPQUEST_API_KEY' with your actual MapQuest API key
mapquest_api_key = 'odRltCJKTn7dKgDHmNm15lNZayE4n4FB'

# Apply the function to calculate driving route for each row
route_info = df.apply(calculate_driving_route, args=(mapquest_api_key,), axis=1)
# print(route_info.head(5))

# Create a new DataFrame with the extracted columns
route_df = pd.DataFrame(list(route_info))
# print(route_df.head(5))

# Concatenate the new DataFrame with the original DataFrame
df = pd.concat([df, route_df], axis=1)
# print(df.head(5))

# Save the updated DataFrame with route information to a CSV file
df.to_csv('categorized_clustered_data_with_route.csv', index=False, sep='|')

#################### function to generate the isochrone #####################################################

#import pandas as pd
#import geopandas as gpd
#from shapely.geometry import Point, Polygon
#import networkx as nx
#import osmnx as ox
#ox.config(log_console=True, use_cache=True)

# Load hub-testing center data
# df = pd.read_csv('categorized_clustered_data_with_route.csv', sep='|')  # Replace with your file path

def get_isochrone(lon, lat, walk_time=10, speed=4.5):
    loc = (lat, lon)
    G = ox.graph_from_point(loc, simplify=True, network_type='walk')
    #G = ox.project_graph(G, to_crs="4483") # Use this line if the coordinates system returned from polys is changed from the original (check which crs you are using)
    gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    x, y = gdf_nodes['geometry'].unary_union.centroid.xy
    center_node = ox.nearest_nodes(G, Y = y[0], X= x[0])
    walking_meters = walk_time * speed * 1000 / 60 #km per hour to m per minute times the minutes to walk
    subgraph = nx.ego_graph(G, center_node, radius=walking_meters, distance='length')
    node_points = [Point(data['x'], data['y']) for node, data in subgraph.nodes(data=True)]
    convex_hull = gpd.GeoSeries(node_points).unary_union.convex_hull
    # Create a Polygon from the node points
#     print(node_points)
#     convex_hull = Polygon([point.coords[0] for point in node_points])
#     print(convex_hull)
#     convex_hull = convex_hull.replace("((", "(").replace("))", ")")
    return convex_hull

data = pd.read_csv('categorized_clustered_data_with_route.csv', sep='|')  # Replace with your file path
    
data['isochrone'] = data.apply(lambda x: get_isochrone(x.stRider_location_long, x.stRider_location_lat), axis=1)
data.to_csv('isochrones1.csv', sep = '|')

################ finally plot the kepler map ################################################

# Load hub-testing center data
data = pd.read_csv('isochrones1.csv', sep='|')  # Replace with your file path
catchment_data = pd.read_csv('catchment_areas_ncr.csv', sep='|')  # Replace with your file path

# Initialize Kepler.gl map
kepler_map = KeplerGl(height=900)

# Add the data to Kepler.gl
kepler_map.add_data(data=data, name='Trips')

# Add the data to Kepler.gl
kepler_map.add_data(data=catchment_data, name='catchment_data')


# Create a color mapping dictionary for UtilizationCategory values
utilization_color_mapping = {
    'Underutilized': '#FFFF00',  # Yellow
    'Overutilized': '#FF0000',   # Red
    'Optimally Utilized': '#00FF00',  # Green
    'Not Operational': '#0000FF'  # Blue
}

# Create a new column 'Color' in your DataFrame based on 'UtilizationCategory'
df['target_icon_Color'] = df['UtilizationCategory'].map(utilization_color_mapping)

# Configure the layers
kepler_map.config = {
    "version": "v1",
    "config": {
        "mapState": {
            "latitude": source_latitude,
            "longitude": source_longitude,
            "zoom": 10
        },
        "mapStyle": {
            "styleType": "dark"
        },
        "visState": {
            "layers": [
                {
                    "id": "Source",
                    "type": "icon",
                    "config": {
                        "dataId": "Trips",
                        "label": "Source",
                        "columns": {
                            "lat": "stRider_location_lat",
                            "lng": "stRider_location_long",
                            "icon": "source_facility_icon"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "radius": 25,
                            "fixedRadius": False,
                            "opacity": 0.8,
                            "color": "#00FF00"
                        }
#                         ,"textLabel": source_tooltip
                    },
                    "visualChannels": {
                        "colorField": None,
                        "colorScale": "quantize",
                        "sizeField": None,
                        "sizeScale": "linear"
                    }
                },
                                {
                    "id": "Catchment Area",
                    "type": "icon",
                    "config": {
                        "dataId": "catchment_data",
                        "label": "catchment Facilities",
                        "columns": {
                            "lat": "Latitude",
                            "lng": "Longitude",
                            "icon": "catchment_area_icon"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "radius": 25,
                            "fixedRadius": False,
                            "opacity": 0.8,
                            "colorRange":{
                                "name": "ColorBrewer PRGn-6",
                                "type": "diverging",
                                "category":"ColorBrewer",
                                "colors": [
                                    '#00FFFF'  # Blue
                                ],
                                "reverse":True
                            }
                        }
#                         ,"textLabel": source_tooltip
                    },
                    "visualChannels": {
                        "colorField": {"name":"Health Facility Type","type" :"real"},
                        "colorScale": "quantize",
                        "sizeField": None,
                        "sizeScale": "linear"
                    }
                },
                {
                    "id": "Target",
                    "type": "icon",
                    "config": {
                        "dataId": "Trips",
                        "label": "Target",
                        "columns": {
                            "lat": "Latitude",
                            "lng": "Longitude",
                            "icon": "target_facility_icon"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "radius": 20,
                            "fixedRadius": False,
                            "opacity": 0.8,
                            "outline": False,
                            "colorRange":{
                                "name": "ColorBrewer PRGn-6",
                                "type": "diverging",
                                "category":"ColorBrewer",
                                "colors": [
                                    '#0000FF',  # Blue
                                    '#FFFF00',  # Yellow
                                    '#FF0000',   # Red
                                    '#00FF00'   # Green
                                ],
                                "reverse":True
                            }
                        }
#                         ,"textLabel": target_tooltip
                    },
                    "visualChannels": {
                        "colorField": {"name":"UtilizationCategory","type" :"real"},
                        "colorScale": "quantize",
                        "sizeField": {"name":"DrivingDistance","type" :"real"},
                        "sizeScale": "quantize"
                    }
                },
                {
                    "id": "Polygons",
                    "type": "geojson",
                    "config": {
                        "dataId": "Trips",
                        "label": "Polygons",
                        "color": ["#FF0000", "#00FF00", "#0000FF"],                        
                        "columns": {
                            "geojson": "route_path_string"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "radius": 20,
                            "opacity": 0.8,
                            "strokeOpacity": 0.8,
                            "thickness": 1,
                            "strokeColor": ["#00FF00", "#0000FF", "#FF0000"],
                          "colorRange": {
                            "name": "Distance travel",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": ["#00FF00", "#0000FF", "#FF0000"]
                          },
                          "strokeColorRange": {
                            "name": "Distance travel",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": ["#00FF00", "#0000FF", "#FF0000"]
                          }
               }
                    },
                   "visualChannels": {
                    "colorField": {"name":"DrivingDistance","type" :"real"},
                    "colorScale": "quantile",
                    "sizeField": None,
                    "sizeScale": "linear",
                    "strokeColorField": {"name":"DrivingDistance","type" :"real"},
                    "strokeColorScale": "quantile",
                    "heightField": None,
                    "heightScale": "linear",
                    "radiusField": None,
                    "radiusScale": "linear"
                  }
                },
                {
                    "id": "isochrones",
                    "type": "geojson",
                    "config": {
                        "dataId": "Trips",
                        "label": "isochrones",
                        "color": ["#FF0000", "#00FF00", "#0000FF"],                        
                        "columns": {
                            "geojson": "isochrone"
                        },
                        "isVisible": True,
                        "visConfig": {
                            "radius": 20,
                            "opacity": 0.8,
                            "strokeOpacity": 0.8,
                            "thickness": 1,
                            "FillColor": ["#00FF00", "#0000FF", "#FF0000"],
                          "colorRange": {
                            "name": "Distance travel",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": ["#00FF00", "#0000FF", "#FF0000"]
                          },
                          "FillColorRange": {
                            "name": "Distance travel",
                            "type": "sequential",
                            "category": "Uber",
                            "colors": ["#00FF00", "#0000FF", "#FF0000"]
                          }
               }
                    },
                   "visualChannels": {
                    "colorField": {"name":"DrivingDistance","type" :"real"},
                    "colorScale": "quantile",
                    "sizeField": None,
                    "sizeScale": "linear",
                    "FillColorField": {"name":"DrivingDistance","type" :"real"},
                    "FillColorScale": "quantile",
                    "heightField": None,
                    "heightScale": "linear",
                    "radiusField": None,
                    "radiusScale": "linear"
                  }
                }
            ],
            "interactionConfig": {
                    "tooltip": {
                      "fieldsToShow": {
                        "Trips": [
                          {
                            "name": "STrider_allocated",
                            "format": None
                          },
                          {
                            "name": "stRider_address",
                            "format": None
                          },
                          {
                            "name": "Disease_type",
                            "format": None
                          },
                          {
                            "name": "Open_From",
                            "format": None
                          },
                          {
                            "name": "Open_To",
                            "format": None
                          },
                          {
                            "name": "Region Name",
                            "format": None
                          },
                          {
                            "name": "Province_name",
                            "format": None
                          },
                          {
                            "name": "City/Municipality Name",
                            "format": None
                          },
                          {
                            "name": "Barangay Name",
                            "format": None
                          },
                          {
                            "name": "Facility Name",
                            "format": None
                          },
                          {
                            "name": "Health Facility Type",
                            "format": None
                          },
                          {
                            "name": "Operational Status",
                            "format": None
                          },
                          {
                            "name": "Total Machines",
                            "format": None
                          },
                          {
                            "name": "Combined Total Modules of ALL GX machines",
                            "format": None
                          },
                          {
                            "name": "Combined Total of Functional Modules",
                            "format": None
                          },
                          {
                            "name": "Combined Total of Non-Functional Modules",
                            "format": None
                          },
                          {
                            "name": "Total_test_completed",
                            "format": None
                          },
                          {
                            "name": "Latitude",
                            "format": None
                          },
                          {
                            "name": "Longitude",
                            "format": None
                          },
                          {
                            "name": "Percentage_utilization",
                            "format": None
                          },
                          {
                            "name": "zscore_calculated",
                            "format": None
                          },
                          {
                            "name": "UtilizationCategory",
                            "format": None
                          },
                          {
                            "name": "DrivingDistance",
                            "textLabel":"Driving Distance in KM",
                            "format": None
                          },
                          {
                            "name": "DrivingTime",
                            "format": None
                          }
                        ],
                        "hex_data": [
                          {
                            "name": "hex_id",
                            "format": None
                          }
                        ]
                      },
                      "compareMode": False,
                      "compareType": "absolute",
                      "enabled": True
                    },
                    "brush": {
                      "size": 0.5,
                      "enabled": False
                    },
                    "geocoder": {
                      "enabled": False
                    },
                    "coordinate": {
                      "enabled": False
                    }
                  },
      "layerBlending": "normal"            
        }
    }
}

config = kepler_map.config

# this will save map with provided data and config
kepler_map.save_to_html(data={'data': data},config=kepler_map.config, file_name='facility_route_network_presentation_final_isochrones.html')

# Display the map
kepler_map


app = Flask(__name__)

@app.route('/')
def index():
    return kepler_map._repr_html_()
    
if __name__ == '__main__':
    app.run(debug=True)  
    
    