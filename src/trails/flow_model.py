import os,sys

from osgeo import gdal
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import igraph as ig
from rasterstats import zonal_stats
import subprocess
import shutil
import seaborn as sns

from multiprocessing import Pool,cpu_count

import pathlib
code_path = (pathlib.Path(__file__).parent.absolute())
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join(code_path,'..','..',"osmconf.ini"))

data_path = os.path.join('..','data')

from simplify import *
from damagescanner.vector import mainRoads,roads
from population_OD import create_bbox,create_grid

pd.options.mode.chained_assignment = None  

def closest_node(node, nodes):
    """[summary]

    Args:
        node ([type]): [description]
        nodes ([type]): [description]

    Returns:
        [type]: [description]
    """    
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def load_network(osm_path,mainroad=True):
    """[summary]

    Args:
        osm_path ([type]): [description]
        mainroad (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    if mainroad:
        df = mainRoads(osm_path)
    else:
        df = roads(osm_path)

    net = Network(edges=df)
    net = clean_roundabouts(net)
    net = split_edges_at_nodes(net)
    net = add_endpoints(net)
    net = add_ids(net)
    net = add_topology(net)    
    net = drop_hanging_nodes(net)    
    net = merge_edges(net)
    net = reset_ids(net) 
    net = add_distances(net)
    net = merge_multilinestrings(net)
    net = fill_attributes(net)
    net = add_travel_time(net)   

    return net

def make_directed(edges):
    
    save_edges = []
    for ind,edge in edges.iterrows():
        if edge.oneway == 'yes':
            save_edges.append(edge)
        else:
            edge.oneway = 'yes'
            edge.lanes = np.round(edge.lanes/2,0)
            save_edges.append(edge)
            edge2 = edge.copy()
            from_id = edge.from_id
            to_id = edge.to_id
            edge2.from_id = to_id
            edge2.to_id = from_id
            save_edges.append(edge2)
            
    new_edges = pd.DataFrame(save_edges).reset_index(drop=True)
    new_edges.id = new_edges.index                
    return new_edges

def get_gdp_values(gdf,data_path):
    """[summary]

    Args:
        gdf ([type]): [description]

    Returns:
        [type]: [description]
    """    
    world_pop = os.path.join(data_path,'global_gdp','GDP_2015.tif')
    gdp = list(item['sum'] for item in zonal_stats(gdf.geometry,world_pop,
                stats="sum"))
    gdp = [x if x is not None else 0 for x in gdp]
    return gdp

def country_grid_gdp_filled(trans_network,country,data_path,rough_grid_split=100,from_main_graph=False):
    """[summary]

    Args:
        trans_network ([type]): [description]
        rough_grid_split (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """    
    if from_main_graph==True:
        node_df = trans_network.copy()
        envelop = shapely.envelope(shapely.multilinestrings(node_df.geometry.values))
        height = np.sqrt(shapely.area(envelop)/rough_grid_split)        
    else:
        node_df = trans_network.nodes.copy()
        node_df.geometry,approximate_crs = convert_crs(node_df)
        envelop = shapely.envelope(shapely.multilinestrings(node_df.geometry.values))
        height = np.sqrt(shapely.area(envelop)/rough_grid_split)    

    gdf_admin = pd.DataFrame(create_grid(create_bbox(node_df),height),columns=['geometry'])

     #load data and convert to pygeos
    country_shape = gpd.read_file(os.path.join(data_path,'GADM','gadm36_levels.gpkg'),layer=0)
    country_shape = pd.DataFrame(country_shape.loc[country_shape.GID_0==country])

    gdf_admin = shapely.intersection(gdf_admin,country_shape.geometry)
    gdf_admin = gdf_admin.loc[~shapely.is_empty(gdf_admin.geometry)]
        
    gdf_admin['centroid'] = shapely.centroid(gdf_admin.geometry)
    gdf_admin['km2'] = area(gdf_admin)
    gdf_admin['gdp'] = get_gdp_values(gdf_admin,data_path)
    gdf_admin = gdf_admin.loc[gdf_admin.gdp > 0].reset_index()
    gdf_admin['gdp_area'] = gdf_admin.gdp/gdf_admin['km2']

    return gdf_admin

def convert_crs(gdf,current_crs="epsg:4326"):
    """[summary]

    Args:
        gdf ([type]): [description]

    Returns:
        [type]: [description]
    """    
    if current_crs == "epsg:4326":
        lat = shapely.geometry.get_y(shapely.centroid(gdf['geometry'].iloc[0]))
        lon = shapely.geometry.get_x(shapely.centroid(gdf['geometry'].iloc[0]))
        # formula below based on :https://gis.stackexchange.com/a/190209/80697         
        approximate_crs = "epsg:" + str(int(32700-np.round((45+lat)/90,0)*100+np.round((183+lon)/6,0)))
    else:
        approximate_crs = "epsg:4326"
        
    #from pygeos/issues/95
    geometries = gdf['geometry']
    coords = shapely.get_coordinates(geometries)
    transformer=pyproj.Transformer.from_crs(current_crs, approximate_crs,always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    result = shapely.set_coordinates(geometries.copy(), np.array(new_coords).T)
    return result,approximate_crs

def area(gdf,km=True):
    """[summary]

    Args:
        gdf ([type]): [description]
        km (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """    
    if km:
        return shapely.area(convert_crs(gdf)[0])/1e6
    else:
        return shapely.area(convert_crs(gdf)[0])

def get_basetable(country,data_path):
    
    io_data_path = os.path.join(data_path,'country_IO_tables')
    
    df = pd.read_csv(os.path.join(io_data_path,'IO_{}_2015_BasicPrice.txt'.format(country)), 
                     sep='\t', skiprows=1,header=[0,1,2],index_col = [0,1,2,3],
                     skipfooter=2617,engine='python')
    
    basetable = df.iloc[:26,:26]
    return basetable.astype(int)

def create_OD(gdf_admin,country_name,data_path):
    """[summary]

    Args:
        gdf_admin ([type]): [description]
        country_name ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # create list of sectors
    sectors = [chr(i).upper() for i in range(ord('a'),ord('o')+1)]
    
    # add a region column if not existing yet.
    if 'NAME_1' not in gdf_admin.columns:
        gdf_admin['NAME_1'] = ['reg'+str(x) for x in list(gdf_admin.index)]

    # prepare paths to downscale a country. We give a country its own directory 
    # to allow for multiple unique countries running at the same time
    downscale_basepath = os.path.join(code_path,'..','..','downscale_od')
    downscale_countrypath = os.path.join(code_path,'..','..','run_downscale_od_{}'.format(country_name))
    
    # copy downscaling method into the country directory
    shutil.copytree(downscale_basepath,downscale_countrypath)

    # save national IO table as basetable for downscaling
    get_basetable(country_name,data_path).to_csv(os.path.join(downscale_countrypath,'basetable.csv'), 
        sep=',',header=False,index=False)

    # create proxy table with GDP values per region/area
    proxy_reg = pd.DataFrame(gdf_admin[['NAME_1','gdp_area']])
    proxy_reg['year'] = 2016
    proxy_reg = proxy_reg[['year','NAME_1','gdp_area']]
    proxy_reg.columns = ['year','id','gdp_area']
    proxy_reg.to_csv(os.path.join(downscale_countrypath,'proxy_reg.csv'),index=False)

    indices = pd.DataFrame(sectors,columns=['sector'])
    indices['name'] = country_name
    indices = indices.reindex(['name','sector'],axis=1)
    indices.to_csv(os.path.join(downscale_countrypath,'indices.csv'),index=False,header=False)    
    
    # prepare yaml file 
    yaml_file = open(os.path.join(downscale_countrypath,"settings_basic.yml"), "r")
    list_of_lines = yaml_file.readlines()
    list_of_lines[6] = '  - id: {}\n'.format(country_name)   
    list_of_lines[8] = '    into: [{}]    \n'.format(','.join(['reg'+str(x) for x in list(gdf_admin.index)]))

    yaml_file = open(os.path.join(downscale_countrypath,"settings_basic.yml"), "w")
    yaml_file.writelines(list_of_lines)
    yaml_file.close()
    
    # run libmrio
    p = subprocess.Popen([os.path.join(downscale_countrypath,'mrio_disaggregate'), 'settings_basic.yml'],
                    cwd=os.path.join(downscale_countrypath))
    
    p.wait()
    
    # create OD matrix from libmrio results
    OD = pd.read_csv(os.path.join(downscale_countrypath,'output.csv'),header=None)
    OD.columns = pd.MultiIndex.from_product([gdf_admin.NAME_1,sectors])
    OD.index = pd.MultiIndex.from_product([gdf_admin.NAME_1,sectors])
    OD = OD.groupby(level=0,axis=0).sum().groupby(level=0,axis=1).sum()
    OD = (OD*5)/365
    OD_dict = OD.stack().to_dict()    
            
    gdf_admin['import'] = list(OD.sum(axis=1))
    gdf_admin['export'] = list(OD.sum(axis=0))
    gdf_admin = gdf_admin.rename({'NAME_1': 'name'}, axis='columns')
    
    # and remove country folder again to avoid clutter in the directory
    shutil.rmtree(downscale_countrypath)

    return OD,OD_dict,sectors,gdf_admin

def prepare_network_routing(transport_network):
    """[summary]

    Args:
        transport_network ([type]): [description]

    Returns:
        [type]: [description]
    """    

    gdf_roads = make_directed(transport_network.edges)
    gdf_roads = gdf_roads.rename(columns={"highway": "infra_type"})
    gdf_roads['GC'] = gdf_roads.apply(gc_function,axis=1)
    gdf_roads['max_flow'] = gdf_roads.apply(set_max_flow,axis=1)
    gdf_roads['flow'] = 0
    gdf_roads['wait_time'] = 0
    
    return gdf_roads

def create_graph(gdf_roads):
    """[summary]

    Args:
        gdf_roads ([type]): [description]

    Returns:
        [type]: [description]
    """    
    gdf_in = gdf_roads.reindex(['from_id','to_id'] + [x for x in list(gdf_roads.columns) if x not in ['from_id','to_id']],axis=1)

    g = ig.Graph.TupleList(gdf_in.itertuples(index=False), edge_attrs=list(gdf_in.columns)[2:],directed=True)
    sg = g.clusters().giant()

    gdf_in.set_index('id',inplace=True)
    
    return sg,gdf_in

def nearest_network_node_list(gdf_admin,gdf_nodes,sg):
    """[summary]

    Args:
        gdf_admin ([type]): [description]
        gdf_nodes ([type]): [description]
        sg ([type]): [description]

    Returns:
        [type]: [description]
    """    
    gdf_nodes = gdf_nodes.loc[gdf_nodes.id.isin(sg.vs['name'])]
    gdf_nodes.reset_index(drop=True,inplace=True)
    nodes = {}
    for admin_ in gdf_admin.itertuples():
        nodes[admin_.name] = gdf_nodes.iloc[shapely.distance((admin_.centroid),gdf_nodes.geometry).idxmin()].id        
    return nodes

def set_max_flow(segment):
    """[summary]

    Args:
        segment ([type]): [description]

    Returns:
        [type]: [description]
    """    
    empty_trip_correction = 0.7 #available capacity for freight reduces
    
    # standard lane capacity = 1000 passenger vehicles per lane per hour
        # trunk and motorway correct by factor 4
        # primary correct by factor 2
        # secondary correct by factor 1
        # tertiary correct factor 0.5
        # other roads correct factor 0.5
    # passenger vehicle equvalent for trucks: 3.5
    # average truck load: 8 tonnes
    # 30 % of trips are empty
    # median value per ton: 2,000 USD
        # median truck value: 8*2000 = 16,000 USD
        
    standard_max_flow = 1000/3.5*16000*empty_trip_correction
    
    if (segment.infra_type == 'trunk') | (segment.infra_type == 'trunk_link'):
        return standard_max_flow*4
    elif (segment.infra_type == 'motorway') | (segment.infra_type == 'motorway_link'):
        return standard_max_flow*4
    elif (segment.infra_type == 'primary') | (segment.infra_type == 'primary_link'):
        return standard_max_flow*2
    elif (segment.infra_type == 'secondary') | (segment.infra_type == 'secondary_link'):
        return standard_max_flow*1
    elif (segment.infra_type == 'tertiary') | (segment.infra_type == 'tertiary_link'):
        return standard_max_flow*0.5
    else:
        return standard_max_flow*0.5
    
def gc_function(segment):
    """[summary]

    Args:
        segment ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # GC = α ∗ WaitT + β ∗ TrvlT + μ ∗ Trate + γ ∗ stddev
    Wait_time = 0
    if segment.infra_type in ['primary','primary_link']:
        Trate = 0.5
        return 0.57*Wait_time+0.49*segment['time']+1*Trate+0.44*1
    elif segment.infra_type in ['secondary','secondary_link']:
        Trate = 1
        return 0.57*Wait_time+0.49*segment['time']+1*Trate+0.44*1
    elif segment.infra_type in ['tertiary','tertiary_link']:
        Trate = 1.5
        return 0.57*Wait_time+0.49*segment['time']+1*Trate+0.44*1
    else:
        Trate = 2
        return 0.57*Wait_time+0.49*segment['time']+1*Trate+0.44*1 
    
def update_gc_function(segment):
    """[summary]

    Args:
        segment ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # GC = α ∗ WaitT + β ∗ TrvlT + μ ∗ Trate + γ ∗ stddev
    
    if segment['flow'] > segment['max_flow']:
        segment['wait_time'] += 1
    elif segment['wait_time'] > 0:
        segment['wait_time'] - 1
    else:    
        segment['wait_time'] = 0
        
    if segment['infra_type'] in ['primary','primary_link']:
        Trate = 0.5
        return 0.57*segment['wait_time']+0.49*segment['time']+1*Trate+0.44*1
    elif segment['infra_type'] in ['secondary','secondary_link']:
        Trate = 1
        return 0.57*segment['wait_time']+0.49*segment['time']+1*Trate+0.44*1
    elif segment['infra_type'] in ['tertiary','tertiary_link']:
        Trate = 1.5
        return 0.57*segment['wait_time']+0.49*segment['time']+1*Trate+0.44*1
    else:
        Trate = 2
        return 0.57*segment['wait_time']+0.49*segment['time']+1*Trate+0.44*1    

def run_flow_analysis(country,transport_network,gdf_admin,OD_dict,notebook=False):
    """[summary]

    Args:
        transport_network ([type]): [description]
        gdf_admin ([type]): [description]

    Returns:
        [type]: [description]
    """    
    plt.rcParams['figure.figsize'] = [5, 5]

    gdf_roads = prepare_network_routing(transport_network)
    sg,gdf_in = create_graph(gdf_roads)
    
    nearest_node = nearest_network_node_list(gdf_admin,transport_network.nodes,sg)
    dest_nodes = [sg.vs['name'].index(nearest_node[x]) for x in list(nearest_node.keys())]

    # this is where the iterations goes
    iterator = 0
    optimal = False
    max_iter = 100
    save_fits = []

    if not notebook:
        plt.ion() ## Note this correction

    # run flow optimization model
    while optimal == False:

        #update cost function per segment, dependent on flows from previous iteration.
        sg.es['GC'] = [(lambda segment: update_gc_function(segment))(segment) for segment in list(sg.es)]
        sg.es['flow'] = 0

        #(re-)assess shortest paths between all regions
        for admin_orig in (list(gdf_admin.name)):
            paths = sg.get_shortest_paths(sg.vs[sg.vs['name'].index(nearest_node[admin_orig])],dest_nodes,weights='GC',output="epath")
            for path,admin_dest in zip(paths,list(gdf_admin.name)):
                flow_value = OD_dict[(admin_orig,admin_dest)]
                sg.es[path]['flow'] = [x + flow_value for x in sg.es[path]['flow']] 

        fitting_edges = (sum([x<y for x,y in zip(sg.es['flow'],sg.es['max_flow'])])/len(sg.es))
        save_fits.append(fitting_edges)

        # if at least 99% of roads are below max flow, we say its good enough
        if (sum([x<y for x,y in zip(sg.es['flow'],sg.es['max_flow'])])/len(sg.es)) > 0.99:
            optimal = True
        iterator += 1

        # when running the code in a notebook, the figure updates instead of a new figure each iteration
        if notebook:
            pl.plot(save_fits) 
            display.display(pl.gcf())
            display.clear_output(wait=True) 
        else:
            plt.plot(save_fits) 
            plt.xlabel('# iteration')
            plt.ylabel('Share of edges below maximum flow')
            plt.show()
            plt.pause(0.0001) #Note this correction

        if iterator == max_iter:
            break    

    # save output
    plt.savefig(os.path.join(code_path,'..','..','figures','{}_flow_modelling.png'.format(country)))   
    gdf_in['flow'] = pd.DataFrame(sg.es['flow'],columns=['flow'],index=sg.es['id'])
    gdf_in['max_flow'] = pd.DataFrame(sg.es['max_flow'],columns=['max_flow'],index=sg.es['id'])
    gdf_in['wait_time'] = pd.DataFrame(sg.es['wait_time'],columns=['wait_time'],index=sg.es['id'])
    gdf_in['overflow'] = gdf_in['flow'].div(gdf_in['max_flow'])   
    
    return gdf_in

def plot_OD_matrix(OD):
    """[summary]

    Args:
        OD ([type]): [description]
    """    
    plt.rcParams['figure.figsize'] = [20, 15]
    sns.heatmap(OD,vmin=0,vmax=1e5,cmap='Reds')

def plot_results(gdf_in):
    """[summary]

    Args:
        gdf_in ([type]): [description]
    """    
    gdf_in['geometry'] = gdf_in.geometry.apply(lambda x : loads(pygeos.to_wkb(x)))

    gdf_plot = gpd.GeoDataFrame(gdf_in)
    gdf_plot.crs = 4326
    gdf_plot = gdf_plot.to_crs(3857)

    plt.rcParams['figure.figsize'] = [20, 10]
    fig, axes = plt.subplots(1, 2)

    for iter_,ax in enumerate(axes.flatten()):
        if iter_ == 0:
            gdf_plot.loc[gdf_plot.flow>1].plot(ax=ax,column='flow',legend=False,cmap='Reds',linewidth=3) #loc[gdf_plot.flow>1]
            ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite,zoom=15)
            ax.set_axis_off()  
            ax.set_title('Flows along the network') 
        else:
            pd.DataFrame(gdf_in.loc[gdf_in.max_flow>1].groupby(
                'infra_type').sum()['distance']/gdf_in.groupby('infra_type').sum()['distance']).dropna().sort_values(
                    by='distance',ascending=False).plot(type='bar',color='red',ax=ax)
            ax.set_ylabel('Percentage of edges > max flow')
            ax.set_xlabel('Road type')

    #plt.show(block=True)    

def country_run(country,data_path=os.path.join('C:\\','Data'),plot=False,save=True):
    """[summary]

    Args:
        country ([type]): [description]
        plot (bool, optional): [description]. Defaults to True.
    """    
    osm_path = os.path.join(data_path,'country_osm','{}.osm.pbf'.format(country))
    
    transport_network = load_network(osm_path)
    print('NOTE: Network created')
    
    gdf_roads = prepare_network_routing(transport_network)
    sg = create_graph(gdf_roads)[0]
    main_graph = pd.DataFrame(list(sg.es['geometry']),columns=['geometry'])
    
    gdf_admin = country_grid_gdp_filled(main_graph,country,data_path,rough_grid_split=100,from_main_graph=True)

    print('NOTE: GDP values extracted')

    # OD,OD_dict,sectors,gdf_admin = create_OD(gdf_admin,country,data_path)
    # print('NOTE: OD created')
    
    # gdf_out = run_flow_analysis(country,transport_network,gdf_admin,OD_dict)
    # print('NOTE: Flow analysis finished')

    # if save:
    #     gdf_admin['geometry'] = gdf_admin.geometry.apply(lambda x: loads(pygeos.to_wkb(x)))
    #     gdf_out = gdf_out.loc[~gdf_out.max_flow.isna()].reset_index(drop=True)
    #     gdf_out_save = gdf_out.copy()
    #     gdf_out_save['geometry'] = gdf_out_save.geometry.apply(lambda x: loads(pygeos.to_wkb(x)))               

    #     gpd.GeoDataFrame(gdf_admin.drop('centroid',axis=1)).to_file(
    #         os.path.join(code_path,'..','..','data',
    #         '{}.gpkg'.format(country)),layer='grid',driver='GPKG')
    #     gpd.GeoDataFrame(gdf_out_save).to_file(os.path.join('..','..','data',
    #         '{}.gpkg'.format(country)),layer='network',driver='GPKG') 

    # if plot:
    #     plot_results(gdf_out) 

if __name__ == '__main__':


    #country_run(sys.argv[1],os.path.join('C:\\','Data'),plot=False)
    #country_run(sys.argv[1],os.path.join(code_path,'..','..','Data'),plot=False)
    #data_path = os.path.join('C:\\','Data') 

    if (len(sys.argv) > 1) & (len(sys.argv[1]) == 3):    
        country_run(sys.argv[1])
    elif (len(sys.argv) > 1) & (len(sys.argv[1]) > 3):    
        glob_info = pd.read_excel(os.path.join('/scistor','ivm','eks510','projects','trails','global_information.xlsx'))
        glob_info = glob_info.loc[glob_info.continent==sys.argv[1]]
        countries = list(glob_info.ISO_3digit)   
        if len(countries) == 0:
            print('FAILED: Please write the continents as follows: Africa, Asia, Central-America, Europe, North-America,Oceania, South-America') 
        with Pool(cpu_count()) as pool: 
          pool.map(country_run,countries,chunksize=1) 
    else:
        print('FAILED: Either provide an ISO3 country name or a continent name')