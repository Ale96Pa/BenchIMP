import os
import pandas as pd
from matplotlib import pyplot as plt
from statistics import median, mean

import config

output_cleanfolder=config.output_cleanfolder
colors_models=config.colors_models
plot_folder=config.plot_folder

def plot_costs():
    cost_df = pd.DataFrame()

    for file_log in os.listdir(output_cleanfolder):
        if "rank" in file_log or "metric" in file_log: continue
        df_log = pd.read_csv(output_cleanfolder+file_log)
        if len(cost_df)==0: cost_df=df_log
        else: cost_df = pd.concat([cost_df, df_log], ignore_index=True)
    
    fig = plt.figure(figsize=(5, 3))
    plt.rcParams.update({'font.size': 14})

    models = []
    for col in cost_df.columns: 
        if "gt" in col: models.append(col)    
    models.sort()


    x_position_increment = 1 / len(models)
    models_labels=[]
    for i in range(0,len(models)):
        positions = []
        positions.append(i)# + (x_position_increment*i))

        values = cost_df[models[i]].to_list()
        norm_val = [float(i)/max(values) for i in values]
        plt.boxplot(norm_val, positions=positions, notch=False, patch_artist=True,
                    boxprops=dict(facecolor=colors_models[i], color="#000000"),
                    capprops=dict(color="#000000"),
                    whiskerprops=dict(color="#000000"),
                    flierprops=dict(color="#000000",
                                    markeredgecolor="#000000"),
                    widths=x_position_increment * 2
                    )
        models_labels.append(models[i].replace("gt","M"))
            
    plt.xlabel("models")
    plt.ylabel("incidents cost")
    plt.xticks(range(0,len(models)), models_labels)
    plt.savefig(plot_folder+"cost_box.png", bbox_inches='tight')
    return

def plot_performance():
    rank_df = pd.DataFrame()

    for dir_ranks in os.listdir(output_cleanfolder):
        if "ranks" not in dir_ranks: continue
        else: 
            for file_ranks in os.listdir(output_cleanfolder+dir_ranks):
                if "weightRank" in file_ranks: 
                    metric_df = pd.read_csv(output_cleanfolder+dir_ranks+"/"+file_ranks)
                    if len(rank_df)==0: rank_df=metric_df
                    else: rank_df = pd.concat([rank_df, metric_df], ignore_index=True)

    fig = plt.figure(figsize=(5, 3))
    plt.rcParams.update({'font.size': 14})

    models = list(set((rank_df["Model"].to_list()+rank_df["Gt"].to_list())))
    models.sort()

    x_position_increment = 1 / len(models)
    models_labels=[]
    for i in range(0,len(models)):
        positions = []
        positions.append(i)# + (x_position_increment*i))

        values = rank_df.query("Model == '" + str(models[i]) + "'")["FinalScore"].to_list()+\
            rank_df.query("Gt == '" + str(models[i]) + "'")["FinalScore"].to_list()
        
        plt.boxplot(values, positions=positions, notch=False, patch_artist=True,
                    boxprops=dict(facecolor=colors_models[i], color="#000000"),
                    capprops=dict(color="#000000"),
                    whiskerprops=dict(color="#000000"),
                    flierprops=dict(color="#000000",
                                    markeredgecolor="#000000"),
                    widths=x_position_increment * 2
                    )
        models_labels.append(models[i].replace("gt","M"))
            
    plt.xlabel("models")
    plt.ylabel("MMR")
    plt.xticks(range(0,len(models)), models_labels)
    plt.savefig(plot_folder+"perfromance_box.png", bbox_inches='tight')
    
    return


def main_analysis():
    plot_costs()
    plot_performance()
