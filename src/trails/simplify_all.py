import os,sys
import pathlib

import shapely
import pandas as pd
import simplify as simply

from osgeo import gdal
from multiprocessing import Pool,cpu_count
from pathlib import Path

from damagescanner.vector import roads

code_path = (pathlib.Path(__file__).parent.absolute())
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join(code_path,'..','..',"osmconf.ini"))

countries = ['ABW', 'AFG', 'AGO', 'AIA', 'ALA', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BES', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CCK', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CUW', 'CXR', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN', 'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUF', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAF', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS', 'MWI', 'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NFK', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'REU', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SHN', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SPM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM', 'SYC', 'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VAT', 'VCT', 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'XAD', 'XCA', 'XKO', 'XNC', 'YEM', 'ZAF', 'ZMB', 'ZWE']
countries = ['DEU', 'FRA', 'USA', 'CHN','RUS']

def filename(country):
    osm_prefix = 'C://Data//kees_pbf/'
    osm_suffix = ".osm.pbf"
    return osm_prefix + country + osm_suffix

if __name__ == '__main__':       
    #data_path = Path("/scistor/ivm/data_catalogue/open_street_map")
    data_path = Path("C://Data//kees_pbf")

    def simp(x):
        print(x)
        roads_to_keep = ['primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','trunk','trunk_link','motorway','motorway_link']
        cGDF = roads(filename(x))
        cGDF = cGDF.loc[cGDF.highway.isin(roads_to_keep)].reset_index(drop=True)
        bob = simply.simplified_network(cGDF)
        a,b = bob.edges,bob.nodes
        a['geometry'] = shapely.to_wkb(a['geometry'])
        b['geometry'] = shapely.to_wkb(b['geometry'])
        pd.to_feather(a,"C://Data//road_networks/"+x+"-edges.feather")
        pd.to_feather(b,"C://Data//road_networks/"+x+"-nodes.feather")
        print(x + " is done")
            #x_ax, isolated_trip_results = net.alt(a)
    
    country = sys.arg[1]
    net_final = simp(country)
