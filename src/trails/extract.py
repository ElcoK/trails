import geopandas
import pandas
import os
import numpy 
from osgeo import ogr
from tqdm import tqdm
from pygeos import from_wkb

def query_b(geoType,keyCol,**valConstraint):
    """
    This function builds an SQL query from the values passed to the retrieve() function.
    Arguments:
         *geoType* : Type of geometry (osm layer) to search for.
         *keyCol* : A list of keys/columns that should be selected from the layer.
         ***valConstraint* : A dictionary of constraints for the values. e.g. WHERE 'value'>20 or 'value'='constraint'
    Returns:
        *string: : a SQL query string.
    """
    query = "SELECT " + "osm_id"
    for a in keyCol: query+= ","+ a  
    query += " FROM " + geoType + " WHERE "
    # If there are values in the dictionary, add constraint clauses
    if valConstraint: 
        for a in [*valConstraint]:
            # For each value of the key, add the constraint
            for b in valConstraint[a]: query += a + b
        query+= " AND "
    # Always ensures the first key/col provided is not Null.
    query+= ""+str(keyCol[0]) +" IS NOT NULL" 
    return query 


def retrieve(osm_path, geoType, keyCol, **valConstraint):
    """
    Function to extract specified geometry and keys/values from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.     
        *geoType* : Type of Geometry to retrieve. e.g. lines, multipolygons, etc.
        *keyCol* : These keys will be returned as columns in the dataframe.
        ***valConstraint: A dictionary specifiying the value constraints.  
        A key can have multiple values (as a list) for more than one constraint for key/value.  
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all columns, geometries, and constraints specified.    
    """
    driver = ogr.GetDriverByName('OSM')
    data = driver.Open(osm_path)
    query = query_b(geoType, keyCol, **valConstraint)
    #q= "SELECT osm_id,highway,oneway,lanes,maxspeed FROM lines WHERE highway='primary' or highway='trunk' or highway='motorway' or highway='motorway_link' or highway='trunk_link' or highway='primary_link' or highway='secondary' or highway='secondary_link' or highway='tertiary' or highway='tertiary_link' AND highway IS NOT NULL"
    #query = q
    sql_lyr = data.ExecuteSQL(query)
    features = []
    # cl = columns 
    cl = ['osm_id']
    cl.extend(keyCol)

    warning_skipped_OSM_features = 0

    if data is not None:
        print('Query is finished, lets start the loop')
        for feature in tqdm(sql_lyr, desc='extract'):
            try:
                if feature.GetField(keyCol[0]) is not None:
                    wkb_geom = feature.geometry().ExportToWkb()
                    geom = from_wkb(bytes(wkb_geom))
                    if geom is None:
                        continue
                    # field will become a row in the dataframe.
                    field = []
                    for i in cl:
                        field.append(feature.GetField(i))
                    field.append(geom)
                    features.append(field)
            except:
                warning_skipped_OSM_features += 1
    else:
        print("ERROR: Nonetype error when requesting SQL. Check required.")

    if warning_skipped_OSM_features > 0:
        print(f"WARNING: skipped {warning_skipped_OSM_features} OSM features")

    cl.append('geometry')

    if len(features) > 0:
        return pandas.DataFrame(features, columns=cl)
    else:
        print("WARNING: No features or No Memory. returning empty GeoDataFrame") 
        return pandas.DataFrame(columns=['osm_id', 'geometry'])


def roads(osm_path):
    """
    Function to extract road linestrings from OpenStreetMap  
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique road linestrings.
    """   
    return retrieve(osm_path, 'lines',['highway','oneway','lanes','maxspeed'])


def roads_bridges(osm_path):
    """
    Function to extract road linestrings from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region
        for which we want to do the analysis.
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique road linestrings.
    """
    return retrieve(osm_path,'lines',['highway','oneway','lanes','maxspeed','bridge','tunnel']) #Changed this for now


def railway(osm_path):
    """
    Function to extract railway linestrings from OpenStreetMap   
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.       
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique land-use polygons.
    """ 
    return retrieve(osm_path,'lines',['railway','service'],**{"service":[" IS NOT NULL"]})

def ferries(osm_path):
    """
    Function to extract road linestrings from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique road linestrings.
    """
    return retrieve(osm_path,'lines',['route'],**{"route":["='ferry'",]})

def electricity(osm_path):
    """
    Function to extract railway linestrings from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique land-use polygons.   
    """    
    return retrieve(osm_path,'lines',['power','voltage'],**{'voltage':[" IS NULL"],})

def mainRoads(osm_path):
    """
    Function to extract main road linestrings from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique main road linestrings.   
    """ 
    return retrieve(osm_path,'lines',['highway','oneway','lanes','maxspeed'],**{'highway':["='primary' or ","='trunk' or ","='motorway' or ","='motorway_link' or ","='trunk_link' or ","='primary_link' or ", "='secondary' or ", "='secondary_link' or ","='tertiary' or ","='tertiary_link'"]})

def primaryRoads(osm_path):
    """
    Function to extract main road linestrings from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region
        for which we want to do the analysis.
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique main road linestrings.
    """
    return retrieve(osm_path,'lines',['highway','oneway','lanes','maxspeed'],**{'highway':["='primary' or ","='trunk' or ","='motorway' or ","='motorway_link' or ","='trunk_link' or ","='primary_link'"]})

def prisecRoads(osm_path):
    """
    Function to extract main road linestrings from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region
        for which we want to do the analysis.
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique main road linestrings.
    """
    return retrieve(osm_path,'lines',['highway','oneway','lanes','maxspeed'],**{'highway':["='primary' or ","='trunk' or ","='motorway' or ","='motorway_link' or ","='trunk_link' or ","='primary_link' or ", "='secondary' or ", "='secondary_link'"]})

def motorwayRoads(osm_path):
    """
    Function to extract main road linestrings from OpenStreetMap
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region
        for which we want to do the analysis.
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique main road linestrings.
    """
    return retrieve(osm_path,'lines',['highway','oneway','lanes','maxspeed'],**{'highway':["='motorway' or ","='motorway_link'"]})

# def preselected_road_types(osm_path,road_types):
#     """
#     Function to extract main road linestrings from OpenStreetMap, for a preselected list of road types
#     Arguments:
#         *osm_path* : file path to the .osm.pbf file of the region
#         for which we want to do the analysis.
#         *road_types* [list of strings]
#     Returns:
#         *GeoDataFrame* : a geopandas GeoDataFrame with all unique main road linestrings.
#     """
#     include_links = True #automatically also create a _link variant per road type
#     new_road_types = []
#
#     if include_links:
#         for road_type in road_types:
#             new_road_types.append(road_type)
#             item = road_type + '_link'
#             new_road_types.append(item)
#     else:
#         new_road_types = [r for r in road_types]
#
#
#
#     highway_list = []
#     for i,road_type in enumerate(new_road_types):
#         item = "='{}'".format(road_type)
#         if not i == len(new_road_types) - 1:
#             item += " and" # Only don't do this for the list item
#         highway_list.append(item)
#
#
#     constraints = {'highway': highway_list}
#
#     retrieve(osm_path, 'lines', ['highway', 'oneway', 'lanes', 'maxspeed'], **{
#         'highway': ["='primary' or ", "='trunk' or ", "='motorway' or ", "='motorway_link' or ", "='trunk_link' or ",
#                     "='primary_link' or ", "='secondary' or ", "='secondary_link'"]})

