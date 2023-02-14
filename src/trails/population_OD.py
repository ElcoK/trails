import os

# set to 0 to use shapely instead of pygeos. Can be removed in the future
os.environ['USE_PYGEOS'] = '0'

import shapely
import geopandas as gpd
import pandas as pd
import numpy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool,cpu_count
from rasterstats import zonal_stats,point_query
from pathlib import Path
import numpy as np

def create_bbox(df):
    """Create bbox around dataframe

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return shapely.box(shapely.total_bounds(df.geometry)[0],
                                  shapely.total_bounds(df.geometry)[1],
                                  shapely.total_bounds(df.geometry)[2],
                                  shapely.total_bounds(df.geometry)[3])

def create_grid(bbox,height):
    """Create a vector-based grid

    Args:
        bbox ([type]): [description]
        height ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # set xmin,ymin,xmax,and ymax of the grid
    xmin, ymin = shapely.total_bounds(bbox)[0],shapely.total_bounds(bbox)[1]
    xmax, ymax = shapely.total_bounds(bbox)[2],shapely.total_bounds(bbox)[3]
    
    #estimate total rows and columns
    rows = int(numpy.ceil((ymax-ymin) / height))
    cols = int(numpy.ceil((xmax-xmin) / height))

    # set corner points
    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    # create actual grid
    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append((
                ((x_left_origin, y_top), (x_right_origin, y_top),
                (x_right_origin, y_bottom), (x_left_origin, y_bottom)
                )))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height

    # return grid as shapely polygons
    return shapely.polygons(res_geoms)


def prepare_possible_OD(x, sindex):
    """Find the nearest node to a given point	

    Args:
        x ([type]): [description]
        sindex ([type]): [description]

    Returns:
        [type]: [description]
    """         
    near = sindex.query_nearest(x,return_distance=True)
    return near[0][0],near[1][0]

def create_final_od_grid(df,height_div,world_pop,country_network):
    """Create a grid of OD points for a given country	

    Args:
        df ([type]): [description]
        height_div ([type]): [description]
        world_pop ([type]): [description]
        country_network ([type]): [description]

    Returns:
        [type]: [description]
    """

    height = numpy.sqrt(shapely.area(df.geometry)/height_div).values[0]
    grid = pd.DataFrame(create_grid(create_bbox(df),height),columns=['geometry'])

    #clip grid of bbox to grid of the actual spatial exterior of the country
    clip_grid = shapely.intersection(grid,df.geometry)
    clip_grid = clip_grid.loc[~shapely.is_empty(clip_grid.geometry)]

    # turn to shapely geometries again for zonal stats        
    clip_grid = gpd.GeoDataFrame(clip_grid)

    if height < 0.01:
        clip_grid.geometry = clip_grid.geometry.centroid
        clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: point_query(x,world_pop)[0])
    else:
        clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: zonal_stats(x,world_pop,stats="sum"))
        clip_grid['tot_pop'] = clip_grid['tot_pop'].apply(lambda x: x[0]['sum'])     
        clip_grid.geometry = clip_grid.geometry.centroid
        
    # remove cells in the grid that have no population data
    clip_grid = clip_grid.loc[~pd.isna(clip_grid.tot_pop)]    
            
    # remove cells with a very low population (most likely also no roads)
    clip_grid = clip_grid.loc[clip_grid.tot_pop > 0] 

    clip_grid.reset_index(inplace=True,drop=True)
    clip_grid['GID_0'] = country_network[:3]
    clip_grid['grid_height'] = height
            
    return clip_grid,height

def create_network_OD_points(country_network):
    """Create a list of OD points for the specified country 

    Args:
        country ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    # set paths to data

    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    data_path = Path(r'C:/data/')
    world_pop = data_path.joinpath('Global_Geospatial','worldpop','ppp_2018_1km_Aggregated.tif')
 
    if data_path.joinpath('Global_Percolation','network_OD_points_revised','{}.csv'.format(country_network[:3])).is_file():
        print("{} already finished!".format(country_network[:3]))           
        return None

            
    #load data and convert to pygeos
    gdf = gpd.read_file('C:\\Data\\natural_earth\\ne_10m_admin_0_countries.shp')
    gdf['ADM0_A3'].loc[gdf.SOVEREIGNT == 'Somaliland'] = 'SOM'
    gdf['ADM0_A3'].loc[gdf.SOVEREIGNT == 'South Sudan'] = 'SSD'
    
    # select country
    country_network = country_network[:3]

    #specify height of cell in the grid     
    upper = 0.95
    lower = 0.05

    #calculate relative area of each country
    gdf['rel_area'] = (shapely.area(gdf.geometry)/shapely.area(gdf.geometry).max())
    gdf['rel_area'].loc[gdf.rel_area > gdf.rel_area.quantile(upper)] = gdf.rel_area.quantile(upper)
    gdf['rel_area'].loc[gdf.rel_area < gdf.rel_area.quantile(lower)] = gdf.rel_area.quantile(lower)

    start = 100
    end = 10000
    width = end - start
    arr = gdf['rel_area'].values
    gdf['grid_size'] = (arr - arr.min())/arr.ptp() * width + start
    gdf['grid_size'] = gdf['grid_size'].astype(int)
    
    #get grid size
    size = gdf.grid_size.loc[gdf.ADM0_A3 == country_network[:3]].values[0]

    #get country ID
    print('{} started with a grid size of {}!'.format(country_network,size))

    # load data
    nodes = pd.read_feather(data_path.joinpath("Global_Percolation","percolation_networks","{}_0-nodes.feather".format(country_network)))
    nodes.geometry = shapely.from_wkb(nodes.geometry)
    nodes.id = ['n_{}'.format(x) for x in nodes.id]
    sindex = shapely.STRtree(nodes['geometry'])
 
    #create dataframe of country row
    geometry = shapely.convex_hull(shapely.multipoints(nodes.geometry.values))
    df = pd.DataFrame([geometry],columns=['geometry'])

    #create grid
    largest_enough_grid = 0
    height_div = size
    while largest_enough_grid < 100:
        clip_grid,height = create_final_od_grid(df,height_div,world_pop,country_network)

        # find nearest node to each cell
        results = pd.DataFrame(clip_grid.geometry.apply(lambda x : prepare_possible_OD(x, 
                                                                                       sindex)).values.tolist(), 
                               columns=['node_id','distance'])

        # merge with grid
        clip_grid = clip_grid.merge(results,left_index=True,right_index=True)

        # remove cells that are too far away from the network
        largest_enough_grid = len(clip_grid.node_id.value_counts())      
        if largest_enough_grid > 100:
            break
        elif largest_enough_grid < 50:
            return None
        else:
            height_div += 50    
    
    # calculate population for each cell
    if height < 0.01:
        node_ids = ['n_{}'.format(x) for x in clip_grid.groupby('node_id').mean().index.values]
        OD_pos = list(zip(node_ids,
                          clip_grid.groupby('node_id').mean().tot_pop.values))
    else:
        node_ids = ['n_{}'.format(x) for x in clip_grid.groupby('node_id').mean().index.values]
        OD_pos = list(zip(node_ids,
                          clip_grid.groupby('node_id').sum().tot_pop.values))        
    
    # create dataframe of OD positions
    df_OD_pos = pd.DataFrame(OD_pos,columns=['node_id','population'])
    
    # save data to file 
    print('{} finished with {} points and {} minimum OD grid!'.format(country_network,len(clip_grid),largest_enough_grid))
    clip_grid.to_csv(data_path.joinpath("Global_Percolation",'network_OD_points_revised','{}.csv'.format(country_network)))
    df_OD_pos.to_csv(data_path.joinpath("Global_Percolation",'OD_pos_revised','{}.csv'.format(country_network)))
    
    return clip_grid,df_OD_pos
    
if __name__ == "__main__":

    # execute only if run as a script
    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    data_path = Path(r'C:/data/Global_Percolation')
    all_files = [ files for files in data_path.joinpath('percolation_networks').iterdir()]
    
    #sorted_files = sorted(all_files, key = os.path.getsize) 
    countries = [y.name[:5] for y in all_files]
    countries = list(set([x for x in countries if '0' in x]))

    create_network_OD_points('TZA')