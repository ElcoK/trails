import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pygeos
import geopandas as gpd
import contextily as cx
import traceback
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def plot_all_atacks():

    # set data paths to results
    data_random_attack = r'C:\Data\percolation_results_random_attack_regular'
    data_random_attack_od_buffer = r'C:\Data\percolation_results_random_attack_od_buffer'
    data_targeted_attack = r'C:\Data\percolation_results_targeted_attack'
    data_local_attack_05 = r'C:\Data\percolation_results_local_attack_05'
    data_local_attack_01 = r'C:\Data\percolation_results_local_attack_01'
    data_local_attack_005 = r'C:\Data\percolation_results_local_attack_005'

    data_path_met= r'C:\Data\percolation_metrics'
    data_path_net = r'C:\Data\percolation_networks'

    data_path_grids = r'C:\Data\percolation_grids'

    # file to get full country names
    glob_info = pd.read_excel(r'C:\Projects\trails\data\global_information.xlsx')

    # get all files from data paths
    perc_files_random_attack = os.listdir(data_random_attack)
    perc_files_random_attack_od_buffer = os.listdir(data_random_attack_od_buffer)
    perc_files_targeted_attack = os.listdir(data_targeted_attack)
    perc_files_local_attack_05 = os.listdir(data_local_attack_05)
    perc_files_local_attack_01 = os.listdir(data_local_attack_01)
    perc_files_local_attack_005 = os.listdir(data_local_attack_005)

    grid_files = os.listdir(data_path_grids)

    met_files = os.listdir(data_path_met)
    net_files = os.listdir(data_path_net)    

    for country in glob_info.ISO_3digit.values:
        network = 0
        #specify file
        file = '{}_{}_results.csv'.format(country,network)
        try:
            # load metrics
            df_metrics = pd.read_csv(os.path.join(data_path_met,[x for x in met_files if file[:5] in x][0]))

            # load percolation results
            df_random = pd.read_csv(os.path.join(data_random_attack,file),index_col=[0])
            df_random_buffer = pd.read_csv(os.path.join(data_random_attack_od_buffer,file),index_col=[0])

            df_random.frac_counter = df_random.frac_counter*100
            df_random_buffer.frac_counter = df_random_buffer.frac_counter*100


            df_target = pd.read_csv(os.path.join(data_targeted_attack,file),index_col=[0])
            df_local_05 = pd.read_csv(os.path.join(data_local_attack_05,file),index_col=[0])
            df_local_01 =pd.read_csv(os.path.join(data_local_attack_01,file),index_col=[0])
            df_local_005 = pd.read_csv(os.path.join(data_local_attack_005,file),index_col=[0])


            # load grids
            grid_05 = pd.read_csv(os.path.join(data_path_grids,'{}_{}_05.csv'.format(country,network)))
            grid_01 = pd.read_csv(os.path.join(data_path_grids,'{}_{}_01.csv'.format(country,network)))
            grid_005 = pd.read_csv(os.path.join(data_path_grids,'{}_{}_005.csv'.format(country,network)))
        except:
            continue

        max_frac_counter = 100#df_random.frac_counter.max()       

        fig, axs = plt.subplots(2,3,figsize=(15,11))

        for iter_,ax in enumerate(axs.flatten()):

            if iter_ == 0:
                sns.boxplot(x="frac_counter", y="pct_isolated", data=df_random,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('Percentage of trips', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of isolated trips', fontsize=15) 
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 

            elif iter_ == 1:
                sns.boxplot(x="frac_counter", y="pct_delayed", data=df_random,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of delayed trips', fontsize=15)      
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 

            elif iter_ == 2:
                sns.boxplot(x="frac_counter", y="pct_unaffected", data=df_random,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of unaffected trips', fontsize=15)      
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))    
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 
                
            elif iter_ == 3:
                sns.boxplot(x="frac_counter", y="pct_isolated", data=df_random_buffer,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('Percentage of trips', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of isolated trips', fontsize=15) 
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 

            elif iter_ == 4:
                sns.boxplot(x="frac_counter", y="pct_delayed", data=df_random_buffer,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of delayed trips', fontsize=15)      
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 

            elif iter_ == 5:
                sns.boxplot(x="frac_counter", y="pct_unaffected", data=df_random_buffer,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                ax.set_ylabel('', fontsize=13)
                ax.set_xlabel('Percentage of network removed', fontsize=13)    
                ax.set_title('Boxplots of unaffected trips', fontsize=15)      
                ax.set_xlim([0, max_frac_counter+2])
                ax.set_ylim([0, 102])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))  
                ax.set_xticks(np.arange(0, max_frac_counter+2, 10)) 
                
        plt.figtext(0.5,0.95, "Random attack", ha="center", va="top", fontsize=18, color="b", fontweight='bold')
        plt.figtext(0.5,0.5, "Random attack with OD buffer", ha="center", va="top", fontsize=18, color="b", fontweight='bold')
        plt.subplots_adjust(hspace = 0.4 )

        plt.suptitle('Main network of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[country]), fontsize=20, fontweight='bold',y=1)        

        plt.savefig(os.path.join(r'C:\Data','figures_random_attack','{}.png'.format(country)),dpi=150)
        plt.clf()

        fig, axs = plt.subplots(4,3,figsize=(15,15))

        for iter_,ax in enumerate(axs.flatten()):
            
            if iter_ == 0:
                isolated_05 = df_local_05.loc[df_local_05.pct_isolated != 0].reset_index(drop=True)
                if len(isolated_05) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No isolated trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
                    
                isolated_05 = isolated_05.sort_values('pct_isolated',ascending=False)
                isolated_05.plot.bar(x='grid_no',y='pct_isolated',ax=ax,legend=False)
                ax.set_xticks([])
                ax.set_title('0.5 degree % isolated trips', fontsize=13) 
                ax.set_xlabel('')
                
            elif iter_ == 1:
                delayed_05 = df_local_05.loc[df_local_05.pct_delayed != 0].reset_index(drop=True)
                if len(delayed_05) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No delayed trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue

                delayed_05 = delayed_05.sort_values('pct_delayed',ascending=False)
                delayed_05.plot.bar(x='grid_no',y='pct_delayed',ax=ax,legend=False)
        
                ax.set_xticks([])
                ax.set_title('0.5 degree % delayed trips', fontsize=13) 
                ax.set_xlabel('')

            elif iter_ == 2:
                ax.axis('off')
                if len(df_local_05) > 0:
                    perc_isolated_trips = round(len(isolated_05)/len(df_local_05)*100,2)
                    perc_delayed_trips = round(len(delayed_05)/len(df_local_05)*100,2)
                    
                    if len(isolated_05) > 0:
                        avg_isolated = round(isolated_05.pct_isolated.mean(),2)   
                    else:
                        avg_isolated = 0
                        
                    if len(delayed_05) > 0:
                        avg_delayed = round(delayed_05.pct_delayed.mean(),2)
                    else:
                        avg_delayed = 0

                else:
                    perc_isolated_trips = 0
                    perc_delayed_trips = 0
                    avg_isolated = 0
                    avg_delayed = 0 
                    
                ax.text(0, 0.8, 'Number of grids in country: {}'.format(len(grid_05)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)            
                ax.text(0, 0.6, 'Number of grids with roads: {}'.format(len(df_local_05)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.5, '% grids causing isolated trips: {}'.format(perc_isolated_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.4, 'Average % of trips isolated: {}'.format(avg_isolated), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)       
                ax.text(0, 0.3, '% grids causing delayed trips: {}'.format(perc_delayed_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.2, 'Average % of trips delayed: {}'.format(avg_delayed), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                
                
            elif iter_ == 3:
                isolated_01 = df_local_01.loc[df_local_01.pct_isolated != 0].reset_index(drop=True)
                if len(isolated_01) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No isolated trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
                
                isolated_01 = isolated_01.sort_values('pct_isolated',ascending=False)
                isolated_01.plot.bar(x='grid_no',y='pct_isolated',ax=ax,legend=False)
                ax.set_xticks([])
                ax.set_title('0.1 degree % isolated trips', fontsize=13) 
                ax.set_xlabel('')
                
            elif iter_ == 4:
                delayed_01 = df_local_01.loc[df_local_01.pct_delayed != 0].reset_index(drop=True)
                if len(delayed_01) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No delayed trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue

                delayed_01 = delayed_01.sort_values('pct_delayed',ascending=False)
                delayed_01.plot.bar(x='grid_no',y='pct_delayed',ax=ax,legend=False)
        
                ax.set_xticks([])
                ax.set_title('0.1 degree % delayed trips', fontsize=13) 
                ax.set_xlabel('')

            elif iter_ == 5:
                ax.axis('off')
                if len(df_local_01) > 0:
                    perc_isolated_trips = round(len(isolated_01)/len(df_local_01)*100,2)
                    perc_delayed_trips = round(len(delayed_01)/len(df_local_01)*100,2)
                    
                    if len(isolated_01) > 0:
                        avg_isolated = round(isolated_01.pct_isolated.mean(),2)   
                    else:
                        avg_isolated = 0
                        
                    if len(delayed_01) > 0:
                        avg_delayed = round(delayed_01.pct_delayed.mean(),2)
                    else:
                        avg_delayed = 0

                else:
                    perc_isolated_trips = 0
                    perc_delayed_trips = 0
                    avg_isolated = 0
                    avg_delayed = 0 
                    
                ax.text(0, 0.8, 'Number of grids in country: {}'.format(len(grid_01)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)            
                ax.text(0, 0.6, 'Number of grids with roads: {}'.format(len(df_local_01)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.5, '% grids causing isolated trips: {}'.format(perc_isolated_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.4, 'Average % of trips isolated: {}'.format(avg_isolated), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)       
                ax.text(0, 0.3, '% grids causing delayed trips: {}'.format(perc_delayed_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.2, 'Average % of trips delayed: {}'.format(avg_delayed), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            
                
            elif iter_ == 6:
                isolated_005 = df_local_005.loc[df_local_005.pct_isolated != 0].reset_index(drop=True)
                if len(isolated_005) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No isolated trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
                    
                isolated_005 = isolated_005.sort_values('pct_isolated',ascending=False)
                isolated_005.plot.bar(x='grid_no',y='pct_isolated',ax=ax,legend=False)
                ax.set_xticks([])
                ax.set_title('0.05 degree % isolated trips', fontsize=13) 
                ax.set_xlabel('')

                
            elif iter_ == 7:
                delayed_005 = df_local_005.loc[df_local_005.pct_delayed != 0].reset_index(drop=True)
                if len(delayed_005) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No delayed trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
                    
                delayed_005 = delayed_005.sort_values('pct_delayed',ascending=False)
                delayed_005.plot.bar(x='grid_no',y='pct_delayed',ax=ax,legend=False)
                    
                ax.set_xticks([])
                ax.set_title('0.05 degree % delayed trips', fontsize=13) 
                ax.set_xlabel('')

            elif iter_ == 8:
                ax.axis('off')
                if len(df_local_005) > 0:
                    perc_isolated_trips = round(len(isolated_005)/len(df_local_005)*100,2)
                    perc_delayed_trips = round(len(delayed_005)/len(df_local_005)*100,2)
                    
                    if len(isolated_005) > 0:
                        avg_isolated = round(isolated_005.pct_isolated.mean(),2)   
                    else:
                        avg_isolated = 0
                        
                    if len(delayed_005) > 0:
                        avg_delayed = round(delayed_005.pct_delayed.mean(),2)
                    else:
                        avg_delayed = 0

                else:
                    perc_isolated_trips = 0
                    perc_delayed_trips = 0
                    avg_isolated = 0
                    avg_delayed = 0 
        
                ax.text(0, 0.8, 'Number of grids in country: {}'.format(len(grid_005)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)            
                ax.text(0, 0.6, 'Number of grids with roads: {}'.format(len(df_local_005)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.5, '% grids causing isolated trips: {}'.format(perc_isolated_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.4, 'Average % of trips isolated: {}'.format(avg_isolated), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)       
                ax.text(0, 0.3, '% grids causing delayed trips: {}'.format(perc_delayed_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.2, 'Average % of trips delayed: {}'.format(avg_delayed), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                
            elif iter_ == 9:
                isolated_target = df_target.loc[df_target.pct_isolated != 0].reset_index(drop=True)
                if len(isolated_target) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No isolated trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
                        
                isolated_target = isolated_target.sort_values('pct_isolated',ascending=False)
                isolated_target.plot.bar(x='edge_no',y='pct_isolated',ax=ax,legend=False)
                ax.set_xticks([])
                ax.set_title('Individual edge % isolated trips', fontsize=13) 
                ax.set_xlabel('')
                
            elif iter_ == 10:
                delayed_target = df_target.loc[df_target.pct_delayed != 0].reset_index(drop=True)
                if len(delayed_target) == 0:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.text(0.5, 0.5, 'No delayed trips!'.format(len(df_target)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,fontweight='bold',fontsize=15)
                    continue
        
                delayed_target = delayed_target.sort_values('pct_delayed',ascending=False)
                delayed_target.plot.bar(x='edge_no',y='pct_delayed',ax=ax,legend=False)
                ax.set_xticks([])
                ax.set_title('Individual edge % delayed trips', fontsize=13) 
                ax.set_xlabel('')

            elif iter_ == 11:
                ax.axis('off')
                if len(df_target) > 0:
                    perc_isolated_trips = round(len(isolated_target)/len(df_target)*100,2)
                    perc_delayed_trips = round(len(delayed_target)/len(df_target)*100,2)
                    
                    if len(isolated_target) > 0:
                        avg_isolated = round(isolated_target.pct_isolated.mean(),2)     
                    else:
                        avg_isolated = 0
                        
                    if len(delayed_target) > 0:
                        avg_delayed = round(delayed_target.pct_delayed.mean(),2)
                    else:
                        avg_delayed = 0

                else:
                    perc_isolated_trips = 0
                    perc_delayed_trips = 0
                    avg_isolated = 0
                    avg_delayed = 0 
                
                ax.text(0, 0.8, 'Number of edges: {}'.format(len(df_target)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.5, '% edges causing isolated trips: {}'.format(perc_isolated_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.4, 'Average % of trips isolated: {}'.format(avg_isolated), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)       
                ax.text(0, 0.3, '% edges causing delayed trips: {}'.format(perc_delayed_trips), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                ax.text(0, 0.2, 'Average % of trips delayed: {}'.format(avg_delayed), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        

        plt.suptitle('Main network of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[country]), fontsize=20, fontweight='bold',y=0.92)
        plt.savefig(os.path.join(r'C:\Data','figures_target_local_attack','{}.png'.format(country)),dpi=150)
        plt.clf()

def plot_percolation_full():

    # set data paths to results
    data_path_perc = r'C:\Data\percolation_results_random_attack_regular'
    data_path_met= r'C:\Data\percolation_metrics'
    data_path_net = r'C:\Data\percolation_networks'

    # file to get full country names
    glob_info = pd.read_excel(r'C:\Projects\trails\data\global_information.xlsx')

    # get all files from data paths
    perc_files = os.listdir(data_path_perc)
    met_files = os.listdir(data_path_met)
    net_files = os.listdir(data_path_net)

    # save the failed ones, so we can check them later
    save_failed = []

    # set x-axis
    x = np.arange(1,100,1)

    # create figure
    fig, axs = plt.subplots(3,2,figsize=(15,20))

    for iter1,file in enumerate(perc_files):
        
        # get name of percolation analysis
        net_name = file[:5]
        
        try:
            if os.path.isfile(os.path.join('..','..','figures','{}_results.png'.format(net_name))):
                print(net_name+" already finished!")
                continue  
                
            # load metrics
            df_metrics = pd.read_csv(os.path.join(data_path_met,[x for x in met_files if file[:5] in x][0]))

            # load percolation results
            df = pd.read_csv(os.path.join(data_path_perc,file),index_col=[0])
            df.frac_counter = df.frac_counter*100
            
            # remove all results where it is pretty much done, so we can zoom onto the interesting part
            df = df.loc[df.pct_isolated < 99.5]   
            max_frac_counter = df.frac_counter.max()       
            
            df_isolated = pd.DataFrame([df.frac_counter.values,df.pct_isolated.values,df.pct_unaffected.values,df.pct_delayed.values]).T
            df_isolated.columns = ['frac_counter','pct_isolated','pct_unaffected','pct_delayed']
            df_sloss = pd.DataFrame([df.frac_counter.values,df.total_pct_surplus_loss_e1.values,df.total_pct_surplus_loss_e2.values]).T
            df_sloss.columns = ['frac_counter','total_pct_surplus_loss_e1','total_pct_surplus_loss_e2']
            
            # load network
            network = pd.read_feather(os.path.join(data_path_net,[x for x in net_files if file[:5] in x][0]))
            network.geometry = pygeos.from_wkb(network.geometry)
            network = gpd.GeoDataFrame(network)
            network.crs = 4326

            # get mean,max,min values
            y_unaff = df_isolated.groupby('frac_counter').max()['pct_unaffected'].values
            y_del = df_isolated.groupby('frac_counter').mean()['pct_delayed'].values
            y_iso = df_isolated.groupby('frac_counter').min()['pct_isolated'].values

            mainnet = 'yes'
            if net_name[4] != '0':
                mainnet = 'no, #{}'.format(net_name)

            if iter1 > 0:
                for iter2,ax in enumerate(axs.flatten()):
                    ax.clear()

            #and plot
            for iter2,ax in enumerate(axs.flatten()):
                
                if iter2 == 0:
                    sns.boxplot(x="frac_counter", y="pct_isolated", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of isolated trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of isolated trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                elif iter2 == 1:
                    network.plot(column='highway',legend=True,ax=ax)
                    try:
                        cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5)
                    except:
                        cx.add_basemap(ax, crs=network.crs.to_string(),alpha=0.5,zoom=10)

                    ax.set_title('Road network', fontsize=15, fontweight='bold')      

                elif iter2 == 2:
                    sns.boxplot(x="frac_counter", y="pct_unaffected", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of unaffected trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of unaffected trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                elif iter2 == 3:
                    sns.boxplot(x="frac_counter", y="pct_delayed", data=df_isolated,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of delayed trips', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)    
                    ax.set_title('Boxplots of delayed trips', fontsize=15, fontweight='bold')      
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                elif iter2 == 4:
                    sns.boxplot(x="frac_counter", y="total_pct_surplus_loss_e1", data=df_sloss,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of surpluss loss (e1)', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)                   
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.set_title('Boxplots of surpluss loss e1', fontsize=15, fontweight='bold')      
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                

                elif iter2 == 5:
                    sns.boxplot(x="frac_counter", y="total_pct_surplus_loss_e2", data=df_sloss,ax=ax,fliersize=0,order=np.arange(max_frac_counter+2),palette="rocket_r",linewidth=0.5)
                    ax.set_ylabel('Percentage of surpluss loss (e2)', fontsize=13)
                    ax.set_xlabel('Percentage of network removed', fontsize=13)                   
                    ax.set_xlim([0, max_frac_counter+2])
                    ax.set_xticks(np.arange(0, max_frac_counter+2, 5)) 
                    ax.set_ylim([0, 102])
                    ax.set_title('Boxplots of surpluss loss e2', fontsize=15, fontweight='bold')      
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                                
                    ax.text(max_frac_counter*0.5, 10.5,'Main Network: {} \nEdges: {} \nDensity: {} \nClique_No: {} \nAssortativity: {} \nDiameter: {} \nMax_Degree: {}'.format( 
                                                                                                mainnet,
                                                                                                    df_metrics.Edge_No.values[0],
                                                                                                    np.round(df_metrics.Density.values[0],7),
                                                                                                    df_metrics.Clique_No.values[0],
                                                                                                    np.round(df_metrics.Assortativity.values[0],7),
                                                                                                    df_metrics.Diameter.values[0],
                                                                                                    df_metrics.Max_Degree.values[0],                                                                                               
                                                                                                    ), fontsize=15)   
                  
            if net_name[4] == '0':    
                if net_name[:3] in ['HKG','TWN','MNP','MAC','MHL','GUM']:
                    name_dict_errors = {'HKG': "Hong Kong",
                    'TWN': "Taiwan",
                    'MNP': "Northern Mariana Islands",
                    'MAC': "Macau",
                    'MHL': "Marshall Islands",
                    'GUM' : "Guam"
                    }
                    plt.suptitle('Main network of {}'.format(name_dict_errors[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
                else:
                    plt.suptitle('Main network of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
            else:
                plt.suptitle('Subnetwork of {}'.format(dict(zip(glob_info.ISO_3digit,glob_info.Country))[net_name[:3]]), fontsize=20, fontweight='bold',y=0.92)
            
            plt.savefig(os.path.join('..','..','figures','{}_results.png'.format(net_name)))
                    
        except Exception as e: 
            print(net_name+" failed because of {}".format(e))
            print(traceback.format_exc())
            save_failed.append(net_name)
        
    print(save_failed)
    
if __name__ == '__main__':       
    plot_all_atacks()
    
