import os
import pandas as pd
from matplotlib import pyplot as plt
from statistics import median, mean

import config

output_cleanfolder=config.output_cleanfolder
colors_models=config.colors_models

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

    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    plt.title('Performance')

    models = list(set((rank_df["Model"].to_list()+rank_df["Gt"].to_list())))
    models.sort()

    x_position_increment = 1 / len(models)
    legend_lines=[]
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
                    widths=(x_position_increment / 3) * 2
                    )
            
    plt.xlabel("Models")
    plt.ylabel("MMR")
    plt.xticks(range(0,len(models)), models) #TODO: replace gt with m
    plt.savefig("test.png") #TODO: path storage
    
    return


def main_analysis():
    plot_performance()

# if __name__ == "__main__":
#     plot_performance()