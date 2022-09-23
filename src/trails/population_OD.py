import os
import pygeos
import geopandas as gpd
import pandas as pd
import numpy
from tqdm import tqdm
import warnings
from shapely.wkb import loads,dumps
warnings.filterwarnings("ignore")
from multiprocessing import Pool,cpu_count
from rasterstats import zonal_stats,point_query
from pathlib import Path
# import feather
import numpy as np

def create_bbox(df):
    """Create bbox around dataframe

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
    """    
    return pygeos.creation.box(pygeos.total_bounds(df.geometry)[0],
                                  pygeos.total_bounds(df.geometry)[1],
                                  pygeos.total_bounds(df.geometry)[2],
                                  pygeos.total_bounds(df.geometry)[3])

def create_grid(bbox,height):
    """Create a vector-based grid

    Args:
        bbox ([type]): [description]
        height ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # set xmin,ymin,xmax,and ymax of the grid
    xmin, ymin = pygeos.total_bounds(bbox)[0],pygeos.total_bounds(bbox)[1]
    xmax, ymax = pygeos.total_bounds(bbox)[2],pygeos.total_bounds(bbox)[3]
    
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

    return pygeos.polygons(res_geoms)

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

    world_pop = data_path.joinpath('worldpop','ppp_2018_1km_Aggregated.tif')

    if data_path.joinpath('network_OD_points','{}.csv'.format(country_network)).is_file():
        return None

        #print("{} already finished!".format(country_network))           
    try: 
        #get country ID
        print('{} started!'.format(country_network))
        
        # load data
        nodes = feather.read_dataframe(data_path.joinpath("percolation_networks","{}-nodes.feather".format(country_network)))
        nodes.geometry = pygeos.from_wkb(nodes.geometry)

        #create dataframe of country row
        geometry = pygeos.convex_hull(pygeos.multipoints(nodes.geometry.values))
        df = pd.DataFrame([geometry],columns=['geometry'])

        #specify height of cell in the grid and create grid of bbox
        
        def create_final_od_grid(df,height_div):
            height = numpy.sqrt(pygeos.area(df.geometry)/height_div).values[0]
            grid = pd.DataFrame(create_grid(create_bbox(df),height),columns=['geometry'])
            
            #clip grid of bbox to grid of the actual spatial exterior of the country
            clip_grid = pygeos.intersection(grid,df.geometry)
            clip_grid = clip_grid.loc[~pygeos.is_empty(clip_grid.geometry)]

            # turn to shapely geometries again for zonal stats
            clip_grid.geometry = pygeos.to_wkb(clip_grid.geometry)
            clip_grid.geometry = clip_grid.geometry.apply(loads)
            clip_grid = gpd.GeoDataFrame(clip_grid)

            # get total population per grid cell

            if height < 0.01:
                clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: point_query(x,world_pop))
                clip_grid['tot_pop'] = clip_grid['tot_pop'].apply(lambda x: np.sum(x))    
            else:
                clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: zonal_stats(x,world_pop,stats="sum"))
                clip_grid['tot_pop'] = clip_grid['tot_pop'].apply(lambda x: x[0]['sum'])               

            # remove cells in the grid that have no population data
            clip_grid = clip_grid.loc[~pd.isna(clip_grid.tot_pop)]
            
            if len(clip_grid) > 100:
                clip_grid = clip_grid.loc[clip_grid.tot_pop > 100]
            clip_grid.reset_index(inplace=True,drop=True)
            clip_grid.geometry = clip_grid.geometry.centroid
            clip_grid['GID_0'] = country_network[:3]
            clip_grid['grid_height'] = height
            #print(len(clip_grid),height)
            return clip_grid
        
        length_clip = 0
        height_div = 200
        save_lengths = []
        while length_clip < 150:
            clip_grid = create_final_od_grid(df,height_div)
            length_clip = len(clip_grid)
            save_lengths.append(length_clip)
            height_div += 50

            if (len(save_lengths) == 6) & (numpy.mean(save_lengths[3:]) < 150):
                break
                
        print('{} finished with {} points!'.format(country_network,len(clip_grid)))
        
        clip_grid.to_csv(data_path.joinpath('network_OD_points','{}.csv'.format(country_network)))
    except:
        None
    
def create_country_OD_points(country):
    """Create a list of OD points for the specified country 

    Args:
        country ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    # set paths to data
    world_pop = r'/scistor/ivm/data_catalogue/open_street_map/worldpop/ppp_2018_1km_Aggregated.tif'
            
    #get country ID
    GID_0 = country['GID_0']
    print('{} started!'.format(GID_0))
    
    #create dataframe of country row
    df = pd.DataFrame(country).T
    df.geometry = pygeos.from_wkb(df.geometry.values[0])
    #specify height of cell in the grid and create grid of bbox
    
    def create_final_od_grid(df,height_div):
        height = numpy.sqrt(pygeos.area(df.geometry)/height_div).values[0]
        grid = pd.DataFrame(create_grid(create_bbox(df),height),columns=['geometry'])

        #clip grid of bbox to grid of the actual spatial exterior of the country
        clip_grid = pygeos.intersection(grid,df.geometry)
        clip_grid = clip_grid.loc[~pygeos.is_empty(clip_grid.geometry)]

        # turn to shapely geometries again for zonal stats
        clip_grid.geometry = pygeos.to_wkb(clip_grid.geometry)
        clip_grid.geometry = clip_grid.geometry.apply(loads)
        clip_grid = gpd.GeoDataFrame(clip_grid)

        # get total population per grid cell
        clip_grid['tot_pop'] = clip_grid.geometry.apply(lambda x: zonal_stats(x,world_pop,stats="sum"))
        clip_grid['tot_pop'] = clip_grid['tot_pop'].apply(lambda x: x[0]['sum'])    

        # remove cells in the grid that have no population data
        clip_grid = clip_grid.loc[~pd.isna(clip_grid.tot_pop)]
        clip_grid = clip_grid.loc[clip_grid.tot_pop > 100]
        clip_grid.reset_index(inplace=True,drop=True)
        clip_grid.geometry = clip_grid.geometry.centroid
        clip_grid['GID_0'] = GID_0
        clip_grid['grid_height'] = height

        return clip_grid
    
    length_clip = 0
    height_div = 200
    save_lengths = []
    while length_clip < 150:
        clip_grid = create_final_od_grid(df,height_div)
        length_clip = len(clip_grid)
        save_lengths.append(length_clip)
        height_div += 50

        if (len(save_lengths) == 6) & (numpy.mean(save_lengths[3:]) < 150):
            break
        
    print('{} finished with {} points!'.format(GID_0,len(clip_grid)))
    
    clip_grid.to_csv(os.path.join(r'/scistor/ivm/data_catalogue/open_street_map/','country_OD_points','{}.csv'.format(GID_0)))
    
    # return the country 
    return clip_grid 

def create_OD_points():
    """[summary]
    """      
    #load data and convert to pygeos
    gdf = gpd.read_file(r'/scistor/ivm/data_catalogue/open_street_map/GADM36/gadm36_levels.gpkg',layer=0)
    tqdm.pandas(desc='Convert geometries to pygeos')
    gdf = pd.DataFrame(gdf)
    gdf['geometry'] = gdf.geometry.progress_apply(dumps)
    
    # create OD points
    save_points_per_country = []
    list_countries = list([x[1] for x in gdf.iterrows()])
    
    #multiprocess all countries
    with Pool(cpu_count()-1) as pool: 
        save_points_per_country = pool.map(create_country_OD_points,list_countries,chunksize=1) 
           
    df_all = pd.concat(save_points_per_country)
    df_all['geometry'] = df_all.geometry.progress_apply(lambda x: pygeos.from_shapely(x))
    df_all.reset_index(inplace=True,drop=True)
    df_all.to_csv('od_points_per_country.csv')
    
if __name__ == "__main__":

    # execute only if run as a script
    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    data_path = Path(r'C:/data/')
    all_files = [ files for files in data_path.joinpath('percolation_networks').iterdir() ]
    sorted_files = sorted(all_files, key = os.path.getsize) 
    countries = [y.name[:5] for y in sorted_files]
    create_network_OD_points('FRA_0')
    #with Pool(8) as pool: 
    #    pool.map(create_network_OD_points,countries,chunksize=1)   

    