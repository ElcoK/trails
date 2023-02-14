import os,sys
import pathlib

import shapely
import pandas as pd
import simplify as simply

from osgeo import gdal
from multiprocessing import Pool,cpu_count
from pathlib import Path

from damagescanner.vector import retrieve

code_path = (pathlib.Path(__file__).parent.absolute())
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join(code_path,'..','..',"osmconf.ini"))

countries = ['ABW', 'AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BES', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CCK', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CUW', 'CXR', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUF', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAF', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS', 'MWI', 'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NFK', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'REU', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SHN', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SPM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM', 'SYC', 'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VAT', 'VCT', 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'XAD', 'XCA', 'XKO', 'XNC', 'YEM', 'ZAF', 'ZMB', 'ZWE']
countries = ['DEU', 'FRA', 'USA', 'CHN','RUS']

def roads(osm_path):
    """
    Function to extract road linestrings from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique road linestrings.
    """   
    return retrieve(osm_path,'lines',['highway','maxspeed','oneway','lanes']) 

def filename(country):
    osm_prefix = "C:\Data\Global_Geospatial\country_osm/"
    osm_suffix = ".osm.pbf"
    return osm_prefix + country + osm_suffix


def simp(country):
    """
    Simplify a road network shapefile for a given country and save the resulting edges and nodes
    as feather files in the "C:/Data/road_networks" directory.

    Args:
        country (str): the name of the country to simplify

    Returns:
        None
    """
    # Define the types of roads to keep
    roads_to_keep = ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 'motorway', 'motorway_link']

    # Load the road network shapefile for the given country and filter out unwanted roads
    cGDF = roads(filename(country))
    cGDF = cGDF.query('highway in @roads_to_keep')
    cGDF = cGDF.reset_index(drop=True)

    # Simplify the road network and extract the resulting edges and nodes
    full_network = simply.simplified_network(cGDF)
    edges, nodes = full_network.edges, full_network.nodes

    # Convert the geometry columns to WKB format for efficient storage as feather files
    edges['geometry'] = shapely.to_wkb(edges['geometry'].values)
    nodes['geometry'] = shapely.to_wkb(nodes['geometry'].values)

    # Save the edges and nodes as feather files
    edges.to_feather(os.path.join("C:/Data/road_networks", f"{country}-edges.feather"))
    nodes.to_feather(os.path.join("C:/Data/road_networks", f"{country}-nodes.feather"))

    # Print a message to indicate that the simplification is complete
    print(f"{country} is done")

if __name__ == '__main__':       

    data_path = Path("C:\Data\Global_Geospatial\country_osm")
    
    country = sys.argv[1]
    net_final = simp(country)
