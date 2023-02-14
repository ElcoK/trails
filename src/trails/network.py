import os,sys

os.environ['USE_PYGEOS'] = '0'

import igraph as ig
import numpy as np

import math
import random
import shapely
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from population_OD import create_bbox,create_grid

data_path = Path(__file__).resolve().parents[2].joinpath('data','percolation')

from multiprocessing import Pool

#import warnings
#warnings.filterwarnings("ignore")

def metrics(graph):
    """This method prints some basic network metrics of an iGraph

    Args:
        graph (iGraph.Graph object): 
    Returns:
        m: 
    """    
    return pd.DataFrame([[graph.ecount(),
    graph.vcount(),
    graph.density(),
    graph.omega(),
    graph.average_path_length(directed=False),
    graph.assortativity_degree(False),
    graph.diameter(directed=False),
    graph.edge_connectivity(),
    graph.maxdegree(),
    graph.transitivity_undirected(),
    len(graph.articulation_points()),
    np.sum(graph.es['distance'])]],
    columns=["Edge_No","Node_No","Density","Clique_No", "Ave_Path_Length", "Assortativity","Diameter","Edge_Connectivity","Max_Degree","Transivitity","Articulation_Points","Total_Edge_Length"])


def metrics_Print(graph):
    """This method prints some basic network metrics of an iGraph

    Args:
        graph (iGraph.Graph object): 
    Returns:
        m: 
    """    
    g = graph
    m = []
    print("Number of edges: ", g.ecount())
    print("Number of nodes: ", g.vcount())
    print("Density: ", g.density())
    print("Number of cliques: ", g.omega())#omega or g.clique_number()
    print("Average path length: ", g.average_path_length(directed=False))
    print("Assortativity: ", g.assortativity_degree(False))
    print("Diameter: ",g.diameter(directed=False))
    print("Edge Connectivity: ", g.edge_connectivity())
    print("Maximum degree: ", g.maxdegree())
    print("Transivitity: ", g.transitivity_local_undirected(weight='time'))
    print("Articulation Points: ", len(g.articulation_points()))    
    print("Total Edge length ", np.sum(g.es['distance']))

#Creates a graph 
def graph_load(edges):
    """Creates 

    Args:
        edges (pandas.DataFrame) : containing road network edges, with from and to ids, and distance / time columns

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """    
    edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
    graph = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)
    graph.vs['id'] = graph.vs['name']
    return graph
    
def graph_load_largest(edges):
    """Returns the largest component of a graph given an edge dataframe

    Args:
        edges (pandas.DataFrame): A dataframe containing from, to ids; time and distance attributes for each edge

    Returns:
        igraph.Graph (object) : a graph with distance and time attributes
    """    
    graph = graph_load(edges)
    return graph.clusters().giant()


def largest_component_df(edges,nodes):
    """Returns the largest component of a network object (network.edges pd  
    and network.nodes pd) with reset ids. Uses igraphs built in function, while adding ids as attributes

    Args:
        edges (pandas.DataFrame): A dataframe containing from and to ids
        nodes (pandas.DataFrame): A dataframe containing node ids

    Returns:
        edges, nodes (pandas.DataFrame) : 2 dataframes containing only those edges and nodes belonging to the giant component
    """    
    edges = edges
    nodes = nodes
    edge_tuples = zip(edges['from_id'],edges['to_id'])
    graph = ig.Graph(directed=False)
    graph.add_vertices(len(nodes))
    graph.vs['id'] = nodes['id']
    graph.add_edges(edge_tuples)
    graph.es['id'] = edges['id']
    graph = graph.clusters().giant()
    edges_giant = edges.loc[edges.id.isin(graph.es()['id'])]
    nodes_giant = nodes.loc[nodes.id.isin(graph.vs()['id'])]
    return reset_ids_network(edges_giant,nodes_giant)


def create_demand(OD_nodes, OD_orig, node_pop):
    """This function creates a demand matrix from the equation:
    
    Demand_a,b = Population_a * Population_b * e^ [-p * Distance_a,b] 
    
    -p is set to 1, populations represent the grid square of the origin, 

    Args:
        OD_nodes (list): a list of nodes to use for the OD, a,b
        OD_orig (np.matrix): A shortest path matrix used for the distance calculation
        node_pop (list): population per OD node

    Returns:
        demand (np.ndarray) : A matrix with demand calculations for each OD pair
    """    
    demand = np.zeros((len(OD_nodes), len(OD_nodes)))

    dist_decay = 1
    maxtrips = 100

    for o in range(0, len(OD_nodes)):
        for d in range(0, len(OD_nodes)):
            if o == d:
                demand[o][d] = 0
            else:
                normalized_dist = OD_orig[o,d] / OD_orig.max()
                demand[o][d] = ((node_pop[o] * node_pop[d]) * np.exp(-1 * dist_decay * normalized_dist))

    demand = ((demand / demand.max()) * maxtrips)
    demand = np.ceil(demand).astype(int)
    return demand

def choose_OD(pos_OD, OD_no):
    """Chooses nodes for OD matrix according to their population size stochastically and probabilistically 

    Args:
        pos_OD (list): a list of tuples representing the nodes and their population
        OD_no (int): Number of OD pairs to create

    Returns:
        OD_nodes [list]: The nodes chosen for the OD
        mapped_pops [list]: Population for nodes chosen
    """    

    #creates 2 tuples of the node ids and their total representative population
    node_ids, tot_pops = zip(*pos_OD)
   
    #Assigns a probability by population size
    pop_probs = [x/sum(tot_pops) for x in tot_pops]

    #OD nodes chosen
    OD_nodes = list(np.random.choice(node_ids, size=OD_no, replace = False, p=pop_probs))

    #Population counts in a mapped list
    node_positions = [node_ids.index(i) for i in OD_nodes]
    mapped_pops = [tot_pops[j] for j in node_positions]

    #returns the nodes, and their populations, should this be zipped?
    return OD_nodes, mapped_pops


def prepare_possible_OD(gridDF, nodes, tolerance = 1):
    """Returns an array of tuples, with the first value the node ID to consider, and the
       second value the total population associated with this node. 
       The tolerance is the size of the bounding box to search for nodes within

    Args:
        gridDF (pandas.DataFrame): A dataframe with the grid centroids and their population
        nodes (pandas.DataFrame): A dataframe of the road network nodes
        tolerance (float, optional): size of the bounding box . Defaults to 0.1.

    Returns:
        final_possible_pop (list): a list of tuples representing the nodes and their population
    """    

    nodeIDs = []
    sindex = shapely.STRtree(nodes['geometry'])

    pos_OD_nodes = []
    pos_tot_pop = []
    for i in gridDF.itertuples():
        ID = nearest(i.geometry, nodes, sindex, tolerance)
        #If a node was found
        #if ID > -1:
        if isinstance(ID,str):
            pos_OD_nodes.append(ID)
            pos_tot_pop.append(i.tot_pop)
            
    a = nodes.loc[nodes.id.isin(pos_OD_nodes)]
    #Create a geopackage of the possible ODs
    #with Geopackage('nodyBGR.gpkg', 'w') as out:
    #    out.add_layer(a, name='finanod', crs='EPSG:4326')
    nodes = np.array([pos_OD_nodes])
    node_unique = np.unique(nodes)
    count = np.array([pos_tot_pop])
    
    #List comprehension to add total populations of recurring nodes 
    final_possible_pop = [(i, count[nodes==i].sum()) for i in node_unique]
    return final_possible_pop

def nearest(geom, gdf,sindex, tolerance):
    """Finds the nearest node

    Args:
        geom (pygeos.Geometry) : Geometry to find nearest
        gdf (pandas.index): Node dataframe to provide possible nodes
        sindex (pygeos.Sindex): Spatial index for faster lookup
        tolerance (float): Size of buffer to use to find nodes

    Returns:
        nearest_geom.id [int]: The node id that is closest to the geom
    """    
    matches_idx = sindex.query(geom)
    if not matches_idx.any():
        buf = shapely.buffer(geom, tolerance)
        matches_idx = sindex.query(buf,'contains').tolist()
    try:
        nearest_geom = min(
            [gdf.iloc[match_idx] for match_idx in matches_idx],
            key=lambda match: shapely.measurement.distance(match.geometry,geom)
        )
    except: 
        #print("Couldn't find node")
        return -1
    return nearest_geom.id

def simple_OD_calc(OD, comparisonOD,pos_trip_no):
    """An alternative OD calculation that counts how many trips exceed threshold length

    Args:
        OD ([type]): [description]
        comparisonOD ([type]): [description]
        pos_trip_no ([type]): [description]

    Returns:
        [type]: [description]
    """    
    compare_thresh = np.greater(OD,comparisonOD)
    over_thresh_no = np.sum(compare_thresh) / 2
    return over_thresh_no / pos_trip_no


def reset_ids_network(edges, nodes):
    """Resets the ids of the nodes and edges, editing 
    the references in edge table using dict masking

    Args:
        edges (pandas.DataFrame): edges to re-reference ids
        nodes (pandas.DataFrame): nodes to re-reference ids

    Returns:
        edges, nodes (pandas.DataFrame) : The re-referenced edges and nodes.
    """    
    nodes = nodes.copy()
    edges = edges.copy()
    to_ids =  edges['to_id'].to_numpy()
    from_ids = edges['from_id'].to_numpy()
    new_node_ids = range(len(nodes))
    #creates a dictionary of the node ids and the actual indices
    id_dict = dict(zip(nodes.id,new_node_ids))
    nt = np.copy(to_ids)
    nf = np.copy(from_ids) 
    #updates all from and to ids, because many nodes are effected, this
    #is quite optimal approach for large dataframes
    for k,v in id_dict.items():
        nt[to_ids==k] = v
        nf[from_ids==k] = v
    edges.drop(labels=['to_id','from_id'],axis=1,inplace=True)
    edges['from_id'] = nf
    edges['to_id'] = nt
    nodes.drop(labels=['id'],axis=1,inplace=True)
    nodes['id'] = new_node_ids
    edges['id'] = range(len(edges))
    edges.reset_index(drop=True,inplace=True)
    nodes.reset_index(drop=True,inplace=True)
    return edges,nodes

def get_metrics_and_split(x):
    
    # try:
    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    #data_path = Path(r'C:/data/')


    if data_path.joinpath("percolation_metrics","{}_0_metrics.csv".format(x)).is_file():
        print("{} already finished!".format(x))           
        return None

    print(x+' has started!')
    edges = pd.read_feather(data_path.joinpath("road_networks","{}-edges.feather".format(x)))
    nodes = pd.read_feather(data_path.joinpath("road_networks","{}-nodes.feather".format(x)))

    #edges = edges.drop('geometry',axis=1)
    edges = edges.reindex(['from_id','to_id'] + [x for x in list(edges.columns) if x not in ['from_id','to_id']],axis=1)
    graph= ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:],directed=False)
    graph.vs['id'] = graph.vs['name']

    #all_df = metrics(graph)
    #all_df.to_csv(data_path.joinpath("percolation_metrics","{}_all_metrics.csv".format(x)))

    cluster_sizes = graph.clusters().sizes()
    cluster_sizes.sort(reverse=True) 
    cluster_loc = [graph.clusters().sizes().index(x) for x in cluster_sizes[:5]]

    main_cluster = graph.clusters().giant()
    main_edges = edges.loc[edges.id.isin(main_cluster.es()['id'])]
    main_nodes = nodes.loc[nodes.id.isin(main_cluster.vs()['id'])]
    main_edges, main_nodes = reset_ids_network(main_edges,main_nodes)
    main_edges.to_feather(data_path.joinpath("percolation_networks","{}_0-edges.feather".format(x)))
    main_nodes.to_feather(data_path.joinpath("percolation_networks","{}_0-nodes.feather".format(x)))
    print(x+' start metrics!')

    main_df = metrics(main_cluster)
    main_df.to_csv(data_path.joinpath("percolation_metrics","{}_0_metrics.csv".format(x)))        
    print(x+' has finished!')


def run_shortest_paths(graph,OD_nodes,weighting='time'):
    
    collect_all_values = []
    for x in (range(len(OD_nodes))):
        get_paths = graph.get_shortest_paths(OD_nodes[x],OD_nodes,weights=weighting,output='epath')

        collect_value = []
        for path in get_paths:
            if len(path) == 0:
                collect_value.append(0.0)
            else:
                collect_value.append(sum(graph.es[path]['time']))

        collect_all_values.append(collect_value)

    return np.matrix(collect_all_values)

def run_shortest_paths_speedup(graph,OD_nodes,weighting='time'):
    
    OD_nodes_sp = OD_nodes.copy()

    collect_all_values = []
    for x in (range(len(OD_nodes))):
        get_paths = graph.get_shortest_paths(OD_nodes_sp[0],OD_nodes_sp,weights=weighting,output='epath')

        collect_value = []
        for path in get_paths:
            if len(path) == 0:
                collect_value.append(0.0)
            else:
                collect_value.append(sum(graph.es[path]['time']))

        del OD_nodes_sp[0]
        collect_all_values.append(np.concatenate((np.zeros(x),np.array(collect_value))))
    
    # make a full matrix by mirroring the triangle
    X = collect_all_values.copy()
    X = np.triu(X)
    X = X + X.T - np.diag(np.diag(X))
        
    return np.matrix(X)

def run_shortest_paths_specific(graph,OD_nodes,trips_to_proceed,default_trips,weighting='time'):


    accounting_matrix = np.zeros((len(OD_nodes),len(OD_nodes)))
    collect_all_values = np.zeros((len(OD_nodes),len(OD_nodes)))
    for x in (range(len(OD_nodes))):
        for y in (range(len(OD_nodes))):
            if accounting_matrix[y,x] == 1:
                collect_all_values[x,y] = collect_all_values[y,x]
                continue

            if trips_to_proceed[x,y]:
                path = graph.get_shortest_paths(OD_nodes[x],OD_nodes[y],weights=weighting,output='epath')[0]
                if len(path) == 0:
                    collect_all_values[x,y] = 0
                else:
                    collect_all_values[x,y] = sum(graph.es[path]['time'])
            else:
                collect_all_values[x,y] = default_trips[x,y]
            
            accounting_matrix[x,y] = 1

    return np.matrix(collect_all_values)


def SummariseOD(OD, fail_value, demand, baseline, GDP_per_capita, frac_counter,distance_disruption, time_disruption):
    """Function returns the % of total trips between origins and destinations that exceed fail value
       Almost verbatim from world bank /GOSTnets world_files_criticality_v2.py

    Args:
        OD (np.matrix): Current OD matrix times (during percolation)
        fail_value (int): Came form GOSTNETS , seems just to be a huge int
        demand (np.ndarray): Demand matrix
        baseline (np.matrix): OD matrix before percolation
        GDP_per_capita (int): GDP of relevant area
        frac_counter (float): Keeps track of current fraction for ease of results storage

    Returns:
        frac_counter, pct_isolated, average_time_disruption, pct_thirty_plus, pct_twice_plus, pct_thrice_plus,total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2
    """
    #adjusted time
    adj_time = OD-baseline

    # total trips
    total_trips = (baseline.shape[0]*baseline.shape[1])-baseline.shape[0]

    #isolated_trips = np.ma.masked_array(masked_demand,~masked_OD.mask)
    isolated_trips_sum = OD[OD == fail_value].shape[1]

    # get percentage of isolated trips
    pct_isolated = (isolated_trips_sum / total_trips)*100

    ## get travel times for remaining trips
    time_unaffected_trips = OD[OD == baseline]

    # get unaffected trips travel times
    if not (np.isnan(np.array(time_unaffected_trips)).all()):
        unaffected_percentiles = []
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),10))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),25))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),50))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),75))
        unaffected_percentiles.append(np.nanpercentile(np.array(time_unaffected_trips),90))
        unaffected_percentiles.append(np.nanmean((time_unaffected_trips)))
    else:
        unaffected_percentiles = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

    # save delayed trips travel times
    delayed_trips_time = adj_time[(OD != baseline) & (np.nan_to_num(np.array(OD),nan=fail_value) != fail_value)]

    unaffected_trips = np.array(time_unaffected_trips).shape[1]
    delayed_trips = np.array(delayed_trips_time).shape[1]

    # save percentage unaffected and delayed
    pct_unaffected = (unaffected_trips/total_trips)*100
    pct_delayed = (delayed_trips/total_trips)*100

    # get delayed trips travel times
    if not (np.isnan(np.array(delayed_trips_time)).all()):

        delayed_percentiles = []
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),10))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),25))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),50))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),75))
        delayed_percentiles.append(np.nanpercentile(np.array(delayed_trips_time),90))
        delayed_percentiles.append(np.nanmean(np.array(delayed_trips_time)))
        average_time_disruption = np.nanmean(np.array(delayed_trips_time))
    else:
        delayed_percentiles = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        average_time_disruption = np.nan
 
    # Flexing demand with trip cost
    def surplus_loss(e, C2, C1, D1):
        """[summary]

        Args:
            e ([type]): [description]
            C2 ([type]): [description]
            C1 ([type]): [description]
            D1 ([type]): [description]

        Returns:
            [type]: [description]
        """
        Y_intercept_max_cost = C1 - (e * D1)

        C2 = np.minimum(C2, Y_intercept_max_cost)

        delta_cost = C2 - C1

        delta_demand = (delta_cost / e)

        D2 = (D1 + delta_demand)

        surplus_loss_ans = ((delta_cost * D2) + ((delta_cost * -delta_demand) / 2))

        triangle = (D1 * (Y_intercept_max_cost - C1) ) / 2

        total_surp_loss = surplus_loss_ans.sum()

        total_pct_surplus_loss = total_surp_loss / triangle.sum()

        return total_surp_loss, total_pct_surplus_loss*100

    adj_cost = (OD * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?
    baseline_cost = (baseline * GDP_per_capita) / (365 * 8 ) #* 3600) time is in hours, so not sure why we do this multiplications with 3600? and minutes would be times 60?

    adj_cost = np.nan_to_num(np.array(adj_cost),nan=np.nanmax(adj_cost))

    total_surp_loss_e1, total_pct_surplus_loss_e1 = surplus_loss(-0.15, adj_cost, baseline_cost, demand)
    total_surp_loss_e2, total_pct_surplus_loss_e2 = surplus_loss(-0.36, adj_cost, baseline_cost, demand)


    return frac_counter, pct_isolated, pct_unaffected, pct_delayed, average_time_disruption, total_surp_loss_e1, total_pct_surplus_loss_e1, total_surp_loss_e2, total_pct_surplus_loss_e2, distance_disruption, time_disruption, unaffected_percentiles, delayed_percentiles


def percolation_random_attack(edges, run_no, country,del_frac=0.01, OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    
    
    #edges.geometry = pyg.from_wkb(edges.geometry)
    print('random attack sample {} started for {}'.format(run_no,country))
    result_df = []
    g = graph_load(edges)
    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list
    edge_no = g.ecount() 
    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    #Creates a matrix of shortest path times between OD nodes
    OD_orig = run_shortest_paths_speedup(g,OD_nodes,weighting='time')
    
    #check which edges are actually being used
    get_paths = []
    for x in (range(len(OD_nodes))):
        get_paths.append(g.get_shortest_paths(OD_nodes[x],OD_nodes,weights='time',output='epath'))

    get_all_path_edges = []
    for path in get_paths:
            get_all_path_edges.append(path)  
            

    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()
    trips_possible = True
    frac_counter = 0 

    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])

    # add frac 0.00 for better figures and results
    result_df.append((0.00, 0, 100, 0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0,0.0, 0.0, 0.0, 0.0, 0.0]))

    while trips_possible:
        if frac_counter > 0.3 and frac_counter <= 0.5: del_frac = 0.02
        if frac_counter > 0.5: del_frac = 0.05
        exp_edge_no = exp_g.ecount()

        #The number of edges to delete
        no_edge_del = max(1,math.floor(del_frac * edge_no))
        try:
            edges_del = random.sample(range(exp_edge_no),no_edge_del)
        except:
            edges_del = range(exp_edge_no)

        exp_g.delete_edges(edges_del)
        frac_counter += del_frac
              
        cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)
        
        #get new matrix
        perc_matrix = run_shortest_paths_speedup(exp_g,OD_nodes,weighting='time')
        np.fill_diagonal(perc_matrix, np.nan)
        perc_matrix[perc_matrix == 0] = 99999999999

        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,round(frac_counter,3),cur_dis_length,cur_dis_time) 
         
        result_df.append(results)
        
        #If the frac_counter goes past 0.99
        if results[0] >= 0.99: break

        #If there are no edges left to remove
        if exp_edge_no < 1: break

    result_df = pd.DataFrame(result_df, columns=['frac_counter', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)
    return result_df

def percolation_random_attack_od_buffer(edges, nodes,grid_height, del_frac=0.01, OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    

    # nodes.geometry = pyg.from_wkb(nodes.geometry)
    # edges.geometry = pyg.from_wkb(edges.geometry)

    result_df = []
    g = graph_load(edges)
    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list
    edge_no = g.ecount() 

    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    buffer_centroids = shapely.buffer(nodes.loc[nodes.id.isin(OD_list)].geometry,grid_height*0.05).values

    OD_buffers = dict(zip(OD_nodes,buffer_centroids))

    edges_per_OD = {}
    for OD_buffer in OD_buffers:
        get_list_edges = list(edges.id.loc[shapely.intersects(shapely.make_valid(OD_buffers[OD_buffer]),shapely.make_valid(edges.geometry.values))].values)     
        edges_per_OD[OD_buffer] = get_list_edges,get_list_edges    

    #Creates a matrix of shortest path times between OD nodes
    OD_orig = run_shortest_paths_speedup(g,OD_nodes,weighting='time')
    OD_thresh = OD_orig * 10
    

    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()
    trips_possible = True

    frac_counter = 0 
    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])


    # add frac 0.00 for better figures and results
    result_df.append((0.00, 0, 100, 0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0,0.0, 0.0, 0.0, 0.0, 0.0]))

    while trips_possible:
        if frac_counter > 0.3 and frac_counter <= 0.5: del_frac = 0.02
        if frac_counter > 0.5: del_frac = 0.05
        exp_edge_no = exp_g.ecount()

        #The number of edges to delete
        no_edge_del = max(1,math.floor(del_frac * edge_no))

        edges_dict = dict(zip(exp_g.es['id'],exp_g.es.indices))    
        edges_dict_reversed = {v: k for k, v in edges_dict.items()}

        try:
            edges_del = random.sample(range(exp_edge_no),no_edge_del)
        except:
            edges_del = range(exp_edge_no)

        #If there are no edges left to remove
        if exp_edge_no < 1: 
            break

        collect_empty_ones = []
        for OD_point in edges_per_OD:
            compared = list(set([edges_dict[x] for x in edges_per_OD[OD_point][1]]) - set(edges_del))
            if len(compared) == 0:
                edges_del = list(set(edges_del).union(set([edges_dict[x] for x in edges_per_OD[OD_point][0]])))
                collect_empty_ones.append(OD_point)
            else:
                edges_del = list(set(edges_del) - (set([edges_dict[x] for x in edges_per_OD[OD_point][0]])))

                edges_per_OD[OD_point] = edges_per_OD[OD_point][0],[edges_dict_reversed[x] for x in compared]

        for e in collect_empty_ones: 
            edges_per_OD.pop(e)

        # only edges around node if all edges are gone
        if (exp_edge_no != 0) | (no_edge_del-len(edges_del) > 0) | (exp_edge_no>len(edges_del)):
            while len(edges_del) < no_edge_del < exp_edge_no:
                edges_del += random.sample(range(exp_edge_no),no_edge_del-len(edges_del))

                collect_empty_ones = []
                for OD_point in edges_per_OD:
                    compared = list(set([edges_dict[x] for x in edges_per_OD[OD_point][1]]) - set(edges_del))
                    if len(compared) == 0:
                        edges_del = list(set(edges_del).union(set([edges_dict[x] for x in edges_per_OD[OD_point][0]])))
                        collect_empty_ones.append(OD_point)
                    else:
                        edges_del = list(set(edges_del) - (set([edges_dict[x] for x in edges_per_OD[OD_point][0]])))

                        edges_per_OD[OD_point] = edges_per_OD[OD_point][0],[edges_dict_reversed[x] for x in compared]

                for e in collect_empty_ones: 
                    edges_per_OD.pop(e)          

          
        exp_g.delete_edges(edges_del)
        frac_counter += del_frac

        cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)

        #get new matrix
        perc_matrix = run_shortest_paths_speedup(exp_g,OD_nodes,weighting='time')
        np.fill_diagonal(perc_matrix, np.nan)
        perc_matrix[perc_matrix == 0] = 99999999999

        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,round(frac_counter,3),cur_dis_length,cur_dis_time) 
        result_df.append(results)

        #If the frac_counter goes past 0.99
        if results[0] >= 0.99: break


    result_df = pd.DataFrame(result_df, columns=['frac_counter', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)
    return result_df

def percolation_targeted_attack(edges,country,network,OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    
    
    result_df = []
    g = graph_load(edges)

    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list
    edge_no = g.ecount() 
    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    #Creates a matrix of shortest path times between OD nodes
    OD_orig = run_shortest_paths_speedup(g,OD_nodes,weighting='time')
    OD_thresh = OD_orig * 10
         
    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()

    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])

    exp_edge_no = g.ecount()

    for edge in tqdm(range(exp_edge_no),total=exp_edge_no,desc='percolation for {} {}'.format(country,network)):
        exp_g = g.copy()
        exp_g.delete_edges(edge)
        
        cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)
        
        #get new matrix
        perc_matrix = run_shortest_paths_speedup(exp_g,OD_nodes,weighting='time')
        np.fill_diagonal(perc_matrix, np.nan)
        perc_matrix[perc_matrix == 0] = 99999999999

        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,g.es[edge]['id'],cur_dis_length,cur_dis_time) 
         
        result_df.append(results)

    result_df = pd.DataFrame(result_df, columns=['edge_no', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)
    return result_df

def percolation_targeted_attack_speedup(edges,edge_sublist,run_no,country,network,OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    
    
    result_df = []
    g = graph_load(edges)


    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list
    edge_no = g.ecount() 

    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    #Creates a matrix of shortest path times between OD nodes
    OD_orig = run_shortest_paths_speedup(g,OD_nodes,weighting='time')
    OD_thresh = OD_orig * 10
    
    #check which edges are actually being used
    get_paths = []
    for x in (range(len(OD_nodes))):
        get_paths.append(g.get_shortest_paths(OD_nodes[x],OD_nodes,weights='time',output='epath'))

    get_all_path_edges = []
    for path in get_paths:
        path = [x for x in path if len(x) > 0]
        #print(path)
        get_all_path_edges.append((path))  

    all_paths_all_nodes = np.zeros((OD_node_no,OD_node_no,10000),dtype='int')

    #create dict of edges per node
    for x in range(len(OD_nodes)):
        for y in range(len(OD_nodes)):
            all_paths_all_nodes[x,y] = np.concatenate((get_paths[y][x],np.zeros((10000-len(get_paths[y][x])),dtype='int')))

    edges_being_used = np.unique([item for sublist in [item for sublist in get_all_path_edges for item in sublist] for item in sublist])   
    edges_being_used = list(set(edges_being_used).intersection(set(edge_sublist.id.values)))

    #print('initial OD created')

    # prepare further analysis
    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()

    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])


    for edge in tqdm(edges_being_used,total=len(edges_being_used),desc='targeted attack part {} for {}'.format(run_no,country)):

        edge_id = g.es[edge]['id']

        # collect only possible trips
        trips_to_proceed = np.zeros((100,100))
        for x in range(len(OD_nodes)):
            for y in range(len(OD_nodes)):
                if len(all_paths_all_nodes[x,y,:][np.in1d(all_paths_all_nodes[x,y,:],[edge_id],assume_unique=True)]) > 0:
                    trips_to_proceed[x,y] = 1
            
        exp_g = g.copy()
        exp_g.delete_edges(edge)
        
        cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
        cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)

        #get new matrix
        perc_matrix = run_shortest_paths_specific(exp_g,OD_nodes,trips_to_proceed,OD_orig,weighting='time')
        #perc_matrix = run_shortest_paths_speedup(exp_g,OD_nodes,weighting='time')

        np.fill_diagonal(perc_matrix, np.nan)
        perc_matrix[perc_matrix == 0] = 99999999999

        results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,g.es[edge]['id'],cur_dis_length,cur_dis_time) 
        
        result_df.append(results)

    result_df = pd.DataFrame(result_df, columns=['edge_no', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)

    return result_df


def percolation_local_attack(edges,df_grid, run_no, country, OD_list=[], pop_list=[], GDP_per_capita=50000):
    """Final version of percolation, runs a simulation on the network provided, to give an indication of network resilience.

    Args:
        edges (pandas.DataFrame): A dataframe containing edge information: the nodes to and from, the time and distance of the edge
        del_frac (float): The fraction to increment the percolation. Defaults to 0.01. e.g.0.01 removes 1 percent of edges at each step
        OD_list (list, optional): OD nodes to use for matrix and calculations.  Defaults to []. 
        pop_list (list, optional): Corresponding population sizes for ODs for demand calculations. Defaults to [].
        GDP_per_capita (int, optional): The GDP of the country/area for surplus cost calculations. Defaults to 50000.

    Returns:
        result_df [pandas.DataFrame]: The results! 'frac_counter', 'pct_isolated', 'average_time_disruption', 'pct_thirty_plus', 'pct_twice_plus', 'pct_thrice_plus','total_surp_loss_e1', 'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2'    """    
    
    result_df = []

    total_runs = len(df_grid)#int(len(df_grid)*0.3)

    # load graph
    g = graph_load(edges)
    
    #These if statements allow for an OD and population list to be randomly generated
    if OD_list == []: 
        OD_nodes = random.sample(range(g.vcount()-1),100)
    else: 
        OD_nodes = OD_list

    OD_node_no = len(OD_nodes)

    if pop_list == []: 
        node_pop = random.sample(range(4000), OD_node_no)
    else:
         node_pop = pop_list

    #Creates a matrix of shortest path times between OD nodes
    OD_orig = run_shortest_paths_speedup(g,OD_nodes,weighting='time')
    
    #check which edges are actually being used
    get_paths = []
    for x in (range(len(OD_nodes))):
        get_paths.append(g.get_shortest_paths(OD_nodes[x],OD_nodes,weights='time',output='epath'))

    get_all_path_edges = []
    for path in get_paths:
        path = [x for x in path if len(x) > 0]
        #print(path)
        get_all_path_edges.append((path))  
       

    all_paths_all_nodes = np.zeros((100,100,10000),dtype='int')
    #create dict of edges per node
    get_all_path_edges_per_node = {}
    for x in range(len(OD_nodes)):
        for y in range(len(OD_nodes)):
            get_all_path_edges_per_node[OD_nodes[x],OD_nodes[y]] = get_paths[y][x]
            all_paths_all_nodes[x,y] = np.concatenate((get_paths[y][x],np.zeros((10000-len(get_paths[y][x])),dtype='int')))

    edges_being_used = np.unique([item for sublist in [item for sublist in get_all_path_edges for item in sublist] for item in sublist])   

    #print('initial OD created')

    # prepare further analysis
    demand = create_demand(OD_nodes, OD_orig, node_pop)
    exp_g = g.copy()
    tot_edge_length = np.sum(g.es['distance'])
    tot_edge_time = np.sum(g.es['time'])

    for k in tqdm(range(total_runs),total=total_runs,desc='local attack part {} for {}'.format(run_no,country)):
        if len(df_grid) == 0:
            break
        
        random_grid = df_grid.sample(n=1)
        
        roads_to_remove = edges.loc[shapely.intersects(edges.geometry.values,random_grid.geometry.values)]
        overlap = list(set(roads_to_remove.id.values).intersection(set(edges_being_used)))
        
        df_grid = df_grid.drop(random_grid.index,axis=0)

        if (len(roads_to_remove) == 0) | (len(overlap) == 0):
            continue   
        else:
            # collect only possible trips
            trips_to_proceed = np.zeros(OD_orig.shape)
            for x in range(len(OD_nodes)):
                for y in range(len(OD_nodes)):
                    if len(all_paths_all_nodes[x,y,:][np.in1d(all_paths_all_nodes[x,y,:],overlap,assume_unique=True)]) > 0:
                        trips_to_proceed[x,y] = 1
                
            exp_g = g.copy()
            exp_g.delete_edges(roads_to_remove.index)
            k += 1          
            cur_dis_length = 1 - (np.sum(exp_g.es['distance'])/tot_edge_length)
            cur_dis_time = 1 - (np.sum(exp_g.es['time'])/tot_edge_time)

            #get new matrix
            perc_matrix = run_shortest_paths_specific(exp_g,OD_nodes,trips_to_proceed,OD_orig,weighting='time')
            #perc_matrix = run_shortest_paths_speedup(exp_g,OD_nodes,weighting='time')
            np.fill_diagonal(perc_matrix, np.nan)
            perc_matrix[perc_matrix == 0] = 99999999999
            results = SummariseOD(perc_matrix, 99999999999, demand, OD_orig, GDP_per_capita,random_grid.index.values[0],cur_dis_length,cur_dis_time) 
            
            result_df.append(results)

    # save results
    result_df = pd.DataFrame(result_df, columns=['grid_no', 'pct_isolated','pct_unaffected', 'pct_delayed',
                                                'average_time_disruption','total_surp_loss_e1', 
                                                'total_pct_surplus_loss_e1', 'total_surp_loss_e2', 'total_pct_surplus_loss_e2',
                                                'distance_disruption','time_disruption','unaffected_percentiles','delayed_percentiles'])
    result_df = result_df.replace('--',0)

    return result_df

def run_percolation_random_attack(country,run_no=200,od_buffer=False,parallel=False):
    """ This function returns results for a single country's transport network.  Possible OD points are chosen
    then probabilistically selected according to the populations each node counts (higher population more likely).

    Args:
        x : country string

    Returns:
        pd.concat(results) (pandas.DataFrame) : The results of the percolation
    """

    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    #data_path = Path(r'C:/data/')

    network = '0'
    #'for network in get_all_networks:
    
    try:
    #     if not od_buffer:
        if data_path.joinpath('percolation_results_random_attack_regular_revised','{}_{}_results.csv'.format(country,network)).is_file():
            print("{} {} already finished!".format(country,network))           
            return None
        else:
            print("Random attack started for {} {}!".format(country,network))  
    

        all_gdp = pd.read_csv(open(data_path.joinpath("percolation_input_data","worldbank_gdp_2019.csv")),error_bad_lines=False)
        gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
        edges = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-edges.feather".format(country,network)))
        nodes = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-nodes.feather".format(country,network)))
        nodes.geometry = shapely.from_wkb(nodes.geometry)

        edges.from_id = ['n_{}'.format(x) for x in edges.from_id]
        edges.to_id = ['n_{}'.format(x) for x in edges.to_id]
        nodes.id = ['n_{}'.format(x) for x in nodes.id]

        # Each country has a set of centroids of grid cells with populations for each cell
        possibleOD = pd.read_csv(open(data_path.joinpath("network_OD_points_revised","{}.csv".format(country,network))))
        grid_height = possibleOD.grid_height.iloc[2]

        seed = sum(map(ord,country))
        random.seed(seed)
        np.random.seed(seed)

        # OD_pos = prepare_possible_OD(possibleOD, nodes, h)

        df_OD_pos = pd.read_csv(open(data_path.joinpath("OD_pos_revised","{}.csv".format(country))))
        OD_pos = list(zip(df_OD_pos.node_id.values,df_OD_pos.population.values))
        OD_no = min(len(OD_pos),100)
        results = []

        if not parallel:
            for x in tqdm(range(run_no),total=run_no,desc='percolation for '+country+' '+network):
                OD_nodes, populations = choose_OD(OD_pos, OD_no)
                if not od_buffer:
                    results.append(percolation_random_attack(edges, 0.01, OD_nodes, populations,gdp))
                else:
                    results.append(percolation_random_attack_od_buffer(edges,nodes,grid_height, 0.01, OD_nodes, populations,gdp))
        else:

            nodes.geometry = shapely.to_wkb(nodes.geometry)
                            
            edges_list = []
            nodes_list = []
            grid_height_list = []
            del_fracs = []
            OD_nodes_list = []
            populations_list = []
            gdp_list = []
            run_list = []
            country_list = []

            for x in range(run_no):
                OD_nodes, populations = choose_OD(OD_pos, OD_no)
                edges_list.append(edges)
                nodes_list.append(nodes)
                grid_height_list.append(grid_height)
                del_fracs.append(0.01)
                OD_nodes_list.append(OD_nodes)
                populations_list.append(populations)
                gdp_list.append(gdp)
                run_list.append(x)
                country_list.append(country)


            with Pool(20) as pool: 
                if not od_buffer:
                    results = pool.starmap(percolation_random_attack,zip(edges_list,run_list,country_list,del_fracs,OD_nodes_list,populations_list,gdp_list),chunksize=1)   
                else:
                    results = pool.starmap(percolation_random_attack_od_buffer,zip(edges_list,nodes_list,grid_height_list,del_fracs,OD_nodes_list,populations_list,gdp_list),chunksize=1)   
                
        res = pd.concat(results)
        if not od_buffer:
            res.to_csv(data_path.joinpath('percolation_results_random_attack_regular_revised','{}_{}_results.csv'.format(country,network)))
        else:
            res.to_csv(data_path.joinpath('percolation_results_random_attack_od_buffer_revised','{}_{}_results.csv'.format(country,network)))

    except Exception as e: 
        print("{} {}  failed because of {}".format(country,network,e))

            
def run_percolation_local_attack(country,run_no=1,grid_size=0.1,parallel=True):
    """ This function returns results for a single country's transport network. Possible OD points are chosen
    then probabilistically selected according to the populations each node counts (higher population more likely).

    Args:
        country : country string

    Returns:
        pd.concat(results) (pandas.DataFrame) : The results of the percolation
    """

    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    #data_path = Path(r'C:/data/')
    network = '0'

    try:
        if data_path.joinpath('percolation_results_local_attack_{}_revised'.format(str(grid_size).replace('.','')),'{}_{}_results.csv'.format(country,network)).is_file():
            print("{} {} already finished!".format(country,network))           
            return None
        
        print("Local attack started for {} {} for {} degrees!".format(country,network,grid_size))  

        all_gdp = pd.read_csv(open(data_path.joinpath("percolation_input_data","worldbank_gdp_2019.csv")),error_bad_lines=False)
        gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
        edges = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-edges.feather".format(country,network)))
        nodes = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-nodes.feather".format(country,network)))
        nodes.geometry = shapely.from_wkb(nodes.geometry)

        print('data loaded for {}'.format(country))

        edges.from_id = ['n_{}'.format(x) for x in edges.from_id]
        edges.to_id = ['n_{}'.format(x) for x in edges.to_id]
        nodes.id = ['n_{}'.format(x) for x in nodes.id]

        # Each country has a set of centroids of grid cells with populations for each cell
        possibleOD = pd.read_csv(open(data_path.joinpath("network_OD_points_revised","{}.csv".format(country))))
        grid_height = possibleOD.grid_height.iloc[2]
        h = grid_height 

        seed = sum(map(ord,country))
        random.seed(seed)
        np.random.seed(seed)

        # OD_pos = prepare_possible_OD(possibleOD, nodes, h)

        df_OD_pos = pd.read_csv(open(data_path.joinpath("OD_pos_revised","{}.csv".format(country))))
        OD_pos = list(zip(df_OD_pos.node_id.values,df_OD_pos.population.values))
        OD_no = min(len(OD_pos),100)
        results = []

        # create grid and determine total_runs
        edges.geometry = shapely.from_wkb(edges.geometry)
        bbox = create_bbox(edges)

        df_grid = pd.DataFrame(create_grid(bbox,grid_size),columns=['geometry'])
        df_grid = df_grid.loc[shapely.intersects(df_grid.geometry.values,shapely.convex_hull(shapely.multilinestrings(edges.geometry.values)))].reset_index(drop=True)
        
        print('grid created for {}'.format(country))

        OD_nodes, populations = choose_OD(OD_pos, OD_no)

        if not parallel:
            results.append(percolation_local_attack(edges,df_grid, OD_nodes, populations,gdp))
        else:
            run_no = 100
            grid_list = np.array_split(df_grid, run_no)
            edges_list = []
            nodes_list = []
            OD_nodes_list = []
            populations_list = []
            gdp_list = []
            runs_list = []
            country_list = []

            for x in range(run_no):
                edges_list.append(edges)
                nodes_list.append(nodes)
                OD_nodes_list.append(OD_nodes)
                populations_list.append(populations)
                gdp_list.append(gdp)
                runs_list.append(x)
                country_list.append(country)

            with Pool(10) as pool: 
                results = pool.starmap(percolation_local_attack,zip(edges_list,grid_list,runs_list,country_list,OD_nodes_list,populations_list,gdp_list),chunksize=1) 

        res = pd.concat(results)
        res.to_csv(data_path.joinpath('percolation_results_local_attack_{}_revised'.format(str(grid_size).replace('.','')),'{}_{}_results.csv'.format(country,network)))

        df_grid.to_csv(data_path.joinpath('percolation_grids','{}_{}_{}.csv'.format(country,network,str(grid_size).replace('.',''))))

    except Exception as e: 
        print("{} {}  failed because of {}".format(country,network,e))

def run_percolation_targeted_attack(country,run_no=10,parallel=True):
    """ This function returns results for a single country's transport network.  Possible OD points are chosen
    then probabilistically selected according to the populations each node counts (higher population more likely).

    Args:
        x : country string

    Returns:
        pd.concat(results) (pandas.DataFrame) : The results of the percolation
    """

    #data_path = Path(r'/scistor/ivm/data_catalogue/open_street_map/')
    #data_path = Path(r'C:/data/')

    #get all networks for a country
    #get_all_networks = [y.name[4] for y in data_path.joinpath("percolation_networks").iterdir() if (y.name.startswith(country) & y.name.endswith('-edges.feather'))]
    network = '0'

    #for network in get_all_networks:
    try:


        if data_path.joinpath('percolation_results_targeted_attack_revised','{}_{}_results.csv'.format(country,network)).is_file():
            print("{} {} already finished!".format(country,network))           
            return None
        
        print("Targeted attack started for {} {}!".format(country,network))  

        all_gdp = pd.read_csv(open(data_path.joinpath("percolation_input_data","worldbank_gdp_2019.csv")),error_bad_lines=False)
        gdp = all_gdp.gdp.loc[all_gdp.iso==country].values[0]
        edges = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-edges.feather".format(country,network)))
        nodes = pd.read_feather(data_path.joinpath("percolation_networks","{}_{}-nodes.feather".format(country,network)))
        nodes.geometry = shapely.from_wkb(nodes.geometry)

        edges.from_id = ['n_{}'.format(x) for x in edges.from_id]
        edges.to_id = ['n_{}'.format(x) for x in edges.to_id]
        nodes.id = ['n_{}'.format(x) for x in nodes.id]

        # Each country has a set of centroids of grid cells with populations for each cell
        possibleOD = pd.read_csv(open(data_path.joinpath("network_OD_points_revised","{}.csv".format(country))))
        grid_height = possibleOD.grid_height.iloc[2]
        h = grid_height 

        seed = sum(map(ord,country))
        random.seed(seed)
        np.random.seed(seed)

        # OD_pos = prepare_possible_OD(possibleOD, nodes, h)

        df_OD_pos = pd.read_csv(open(data_path.joinpath("OD_pos_revised","{}.csv".format(country))))
        OD_pos = list(zip(df_OD_pos.node_id.values,df_OD_pos.population.values))
        OD_no = min(len(OD_pos),100)
        results = []
        
        #for x in range(run_no):#,total=run_no,desc='percolation for '+country+' '+network):
        OD_nodes, populations = choose_OD(OD_pos, OD_no)

        if not parallel:
            results.append(percolation_targeted_attack_speedup(edges,country,network,OD_nodes,populations,gdp))
        else:

            run_no = 100
            edge_sublist_list = np.array_split(edges, run_no)
            edges_list = []
            nodes_list = []
            OD_nodes_list = []
            populations_list = []
            country_list = []
            network_list = []
            gdp_list = []
            runs_list = []


            for x in range(run_no):
                edges_list.append(edges)
                nodes_list.append(nodes)
                OD_nodes_list.append(OD_nodes)
                populations_list.append(populations)
                gdp_list.append(gdp)
                runs_list.append(x)
                country_list.append(country)
                network_list.append(country)

            with Pool(20) as pool: 

                results = pool.starmap(percolation_targeted_attack_speedup,zip(edges_list,
                                                                                edge_sublist_list,
                                                                                runs_list,
                                                                                country_list,
                                                                                network_list,
                                                                                OD_nodes_list,
                                                                                populations_list,
                                                                                gdp_list),chunksize=1) 


        res = pd.concat(results)
        res.to_csv(data_path.joinpath('percolation_results_targeted_attack_revised','{}_{}_results.csv'.format(country,network)))

    except Exception as e: 
        print("{} {}  failed because of {}".format(country,network,e))

def run_random_attack_percolations(country):

    # random attack
    run_percolation_random_attack(country,run_no=200,od_buffer=False,parallel=True)
   
def run_local_targeted_attack_percolations(country):

    # local attack
    run_percolation_local_attack(country,grid_size=0.5)
    run_percolation_local_attack(country,grid_size=0.1)
    run_percolation_local_attack(country,grid_size=0.05)

    # targeted attack  
    run_percolation_targeted_attack(country)

if __name__ == '__main__':     

    #data_path = Path("/scistor/ivm/data_catalogue/open_street_map")
    data_path = Path(r'C:/data/Global_Percolation')

    to_ignore = ['ICA', 'SPM', 'XPI', 'SJM', 'ALA', 'TUV', 'PCN', 'XNC', 'IOT', 'ATF', 'XCA', # initial list, below are the ones smaller than 100 nodes
    'GRL', 'GRD', 'SLB', 'FLK', 'MSR', 'NFK', 'VCT', 'VUT', 'TCA', 'NRU', 'WSM', 'VGB', 'VAT', 
    'BLM', 'BES', 'FSM', 'MAF', 'WLF', 'ASM', 'NIU', 'CXR', 'AIA', 'PLW','DMA', 'MDV', 'COK', 'KIR', 'MHL', 'SHN','CCK']

    all_files = [ files for files in data_path.joinpath('network_OD_points_revised').iterdir() ]
    sorted_files = sorted(all_files, key = os.path.getsize) 
    countries = [y.name[:3] for y in sorted_files]
    

    country = sys.argv[1]
    # # #print(countries[int(sys.argv[1]]))
    # if (country in fin_countries) | (country in to_ignore):
    #     print('{} already done or too small!'.format(country))

    #run_local_targeted_attack_percolations(sys.argv[1])
    #run_percolation_local_attack(sys.argv[1],grid_size=np.float(sys.argv[2]))
    # # run_percolation_targeted_attack('NLD')
    # # #run_percolation_per_grid(sys.argv[1],run_no=1)
    #run_random_attack_percolations('AND')

    # for country in countries:
    run_random_attack_percolations(country)   
    #run_random_attack_percolations(country)    


    #with Pool(15) as pool: 
    #    pool.map(run_local_targeted_attack_percolations,countries,chunksize=1)   
