import utm
import numpy as np
import pandas as pd
import time
import censusgeocode as cg
from tqdm import tqdm

# Define a function creating smaller bounding boxes
def create_bbox(w, lat_min, lon_min, lat_max, lon_max):
    '''
    Create a dataframe where each column represents latitude and longitude of
    southern, western, northern, and eastern edge points of an inner bounding box
    as well as the center point of the bounding boxes

    Parameters
    ----------
    w: width (miles) of a bounding box (inner square)
    lat_min: minimum latitude of the whole boundary
    lon_min: minimum longitude of the whole boundary
    lat_max: maximum latitude of the whole boundary
    lon_max: maximum longitude of the whole boundary

    '''

    # Convert latitude and longitude to UTM coordinate
    easting_init, northing_init, zone_number_init, zone_letter_init = utm.from_latlon(lat_min, lon_min)
    easting_end, northing_end, zone_number_end, zone_letter_end = utm.from_latlon(lat_max, lon_max)

    # Convert the unit of bbox width and height to meters
    bbox_width_meter, bbox_height_meter = 1609.344 * w, 1609.344 * w

    #if (zone_number_init == zone_number_end) and (zone_letter_init == zone_letter_end):

    # Find the bbox's latitude and longitude
    # for southern, western, northern, and eastern edge
    south = [northing_init]
    west = [easting_init]

    bbox_south = northing_init
    bbox_west = easting_init

    while bbox_south <= northing_end:
        bbox_south += bbox_height_meter
        south.append(bbox_south)

    while bbox_west <= easting_end:
        bbox_west += bbox_width_meter
        west.append(bbox_west)

    south_edge = []
    west_edge = []
    north_edge = []
    east_edge = []
    center_point_x = []
    center_point_y = []
    row_num = []
    col_num = []

    for i in range(len(south) - 1):

        for j in range(len(west) - 1):
            row_num.append(i)
            col_num.append(j)
            south_edge.append(south[i])
            west_edge.append(west[j])
            north_edge.append(south[i + 1])
            east_edge.append(west[j + 1])
            center_point_x.append((west[j] + west[j + 1]) / 2)
            center_point_y.append((south[i] + south[i + 1]) / 2)

    df_bbox = pd.DataFrame({'row_num': row_num,
                            'col_num': col_num,
                            'south_edge': south_edge,
                            'west_edge': west_edge,
                            'north_edge': north_edge,
                            'east_edge': east_edge,
                            'center_x': center_point_x,
                            'center_y': center_point_y, })

    # else:
    #     print("error: zone_number and zone_letter are not consistent")

    # Convert utm back to latitude and longitude
    df_bbox['south_west_lat_long'] = df_bbox[['south_edge', 'west_edge']].apply(lambda df_bbox:
                                                                                utm.to_latlon(df_bbox['west_edge'],
                                                                                              df_bbox['south_edge'],
                                                                                              zone_number_init,
                                                                                              zone_letter_init), axis=1)
    df_bbox['north_east_lat_long'] = df_bbox[['north_edge', 'east_edge']].apply(lambda df_bbox:
                                                                                utm.to_latlon(df_bbox['east_edge'],
                                                                                              df_bbox['north_edge'],
                                                                                              zone_number_init,
                                                                                              zone_letter_init), axis=1)
    df_bbox['center_latlon'] = df_bbox[['center_x', 'center_y']].apply(lambda df_bbox:
                                                                       utm.to_latlon(df_bbox['center_x'],
                                                                                     df_bbox['center_y'],
                                                                                     zone_number_init,
                                                                                     zone_letter_init), axis=1)

    # Return the final df format
    df_bbox['swne_edges'] = df_bbox['south_west_lat_long'] + df_bbox['north_east_lat_long']
    columns = ['row_num', 'col_num', 'swne_edges', 'center_latlon']
    return df_bbox[columns]


# Create inner bounding boxes for San Francisco 
df_bbox = create_bbox(w=.1, lat_min=37.704744, lon_min=-122.511930, lat_max=37.837586, lon_max=-122.356061)

# Extract Census Tract GEOID using latitude and longitude (center location of each bbox)
df_bbox['center_lat'] = df_bbox['center_latlon'].str[0]
df_bbox['center_lon'] = df_bbox['center_latlon'].str[1]

# Extract GEOID for census block (remove the last three digits of GEOID for census block group)
df_bbox['GEOID'] = np.nan

for idx, chunk in tqdm(enumerate(np.array_split(df_bbox, 100))):
    for i in chunk.index:
        geoid = cg.coordinates(chunk['center_lon'][i], chunk['center_lat'][i])['2020 Census Blocks'][0]['GEOID'][:-3]
        df_bbox['GEOID'][i] = geoid
    df_bbox.to_csv(f'data/df_bbox_{idx}.csv')

# # drop tuple object columns
# df_bbox = df_bbox.drop(['swne_edges','center_latlon','coords'], axis=1)

# # Save df_bbox as a geojson file
# df_bbox.to_file('data/df_bbox_2020.geojson', driver="GeoJSON")