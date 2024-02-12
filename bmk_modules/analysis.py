import os
import pandas as pd
from matplotlib import pyplot as plt
from statistics import median, mean

import config

output_cleanfolder=config.output_cleanfolder
output_noisedresult_folder=config.output_noisedresult_folder
colors_models=config.colors_models
plot_folder=config.plot_folder

noise_magnitude_step=config.magnitude_step
typology_noising_step = config.ratio_step
features=config.features_list

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
        
        if "ETR" in models[i]: models_labels.append(models[i].replace("gt",""))
        else: models_labels.append(models[i].replace("gt","M"))
            
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
        norm_val = [1-float(i) for i in values]
        
        plt.boxplot(norm_val, positions=positions, notch=False, patch_artist=True,
                    boxprops=dict(facecolor=colors_models[i], color="#000000"),
                    capprops=dict(color="#000000"),
                    whiskerprops=dict(color="#000000"),
                    flierprops=dict(color="#000000",
                                    markeredgecolor="#000000"),
                    widths=x_position_increment * 2
                    )
        if "ETR" in models[i]: models_labels.append(models[i].replace("gt",""))
        else: models_labels.append(models[i].replace("gt","M"))
            
    plt.xlabel("models")
    plt.ylabel("MMR")
    plt.xticks(range(0,len(models)), models_labels)
    plt.savefig(plot_folder+"perfromance_box.png", bbox_inches='tight')
    
    return

def plot_single_err(err_type):
    rank_df = pd.DataFrame()

    for dir_ranks in os.listdir(output_cleanfolder):
        if "ranks" not in dir_ranks: continue
        else: 
            for file_ranks in os.listdir(output_cleanfolder+dir_ranks):
                if err_type in file_ranks: 
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

        if err_type == "mae":
            values = rank_df.query("Model == '" + str(models[i]) + "'")["mae_n"].to_list()+\
            rank_df.query("Gt == '" + str(models[i]) + "'")["mae_n"].to_list()
        elif err_type == "mse":
            values = rank_df.query("Model == '" + str(models[i]) + "'")["mse_n"].to_list()+\
            rank_df.query("Gt == '" + str(models[i]) + "'")["mse_n"].to_list()
        else:
            values = rank_df.query("Model == '" + str(models[i]) + "'")["median_err_n"].to_list()+\
            rank_df.query("Gt == '" + str(models[i]) + "'")["median_err_n"].to_list()
        
        plt.boxplot(values, positions=positions, notch=False, patch_artist=True,
                    boxprops=dict(facecolor=colors_models[i], color="#000000"),
                    capprops=dict(color="#000000"),
                    whiskerprops=dict(color="#000000"),
                    flierprops=dict(color="#000000",
                                    markeredgecolor="#000000"),
                    widths=x_position_increment * 2
                    )
        if "ETR" in models[i]: models_labels.append(models[i].replace("gt",""))
        else: models_labels.append(models[i].replace("gt","M"))
            
    plt.xlabel("models")
    plt.ylabel(err_type.replace("median","mad"))
    plt.xticks(range(0,len(models)), models_labels)
    plt.savefig(plot_folder+err_type+"_perfromance_box.png", bbox_inches='tight')
    
    return

# Define a class for holding data related to the robustness of models and typology
class model_typology_robustness_data:
    def __init__(self, models, typology_noising_step):
        self.models_magnitude = []
        for model in models:
            for typology_step in range(typology_noising_step, 100 + typology_noising_step, typology_noising_step):
                self.models_magnitude.append((model, typology_step))
        self.robustness = []
        self.agg_robustness = []
        for model_magnitude in self.models_magnitude:
            self.robustness.append([])
            self.agg_robustness.append([])

    def add_robustness(self, model_megnitude, robustness_data):
        index_model = self.models_magnitude.index(model_megnitude)
        self.robustness[index_model] += robustness_data

    def compute_aggregated_robustness(self):
        for x in range(len(self.models_magnitude)):
            self.agg_robustness[x] = mean(self.robustness[x])

def select_fixed_magnitude_paths(root, magnitude):
    files = os.listdir(root)
    return list(filter(lambda x: f"P{magnitude}_[" in x, files))

def filter_fixed_typology(files, typology, magnitude):
    if typology == 'null_values':
        return list(filter(lambda x: f"V{magnitude}" in x, files))
    elif typology == 'incorrect':
        return list(filter(lambda x: f"N{magnitude}" in x, files))
    elif typology == 'imprecise':
        return list(filter(lambda x: f"M{magnitude}" in x, files))

def read_clean_classification_files(clean_classification_path, metric):
    if metric == 'mse':
        return pd.read_csv(f"{clean_classification_path}/mseRank.csv")
    elif metric == 'mae':
        return pd.read_csv(f"{clean_classification_path}/maeRank.csv")
    elif metric == 'median':
        return pd.read_csv(f"{clean_classification_path}/medianRank.csv")
    else:
        result = [
            pd.read_csv(f"{clean_classification_path}/mseRank.csv"),
            pd.read_csv(f"{clean_classification_path}/maeRank.csv"),
            pd.read_csv(f"{clean_classification_path}/medianRank.csv")
        ]
        return pd.concat(result)
    
def read_classification_files(root, metric):
    if not os.path.exists(f"{root}/ranks/mseRank.csv"): return pd.DataFrame()
    if metric == 'mse':
        return pd.read_csv(f"{root}/ranks/mseRank.csv")
    elif metric == 'mae':
        return pd.read_csv(f"{root}/ranks/maeRank.csv")
    elif metric == 'median':
        return pd.read_csv(f"{root}/ranks/medianRank.csv")
    else:
        result = [
            pd.read_csv(f"{root}/ranks/mseRank.csv"),
            pd.read_csv(f"{root}/ranks/maeRank.csv"),
            pd.read_csv(f"{root}/ranks/medianRank.csv")
        ]
        return result

def plot_robustness():
    metrics=["mae", "mse", "median"]

    possible_noise_magnitude = []
    for noise_magnitude in range(noise_magnitude_step, 51, noise_magnitude_step):
        possible_noise_magnitude.append(noise_magnitude)
        
    for noise_magnitude in possible_noise_magnitude:
        for metric in metrics:
            for typology in ["incorrect", "imprecise", "null_values"]:
                clean_dataset = read_clean_classification_files(output_cleanfolder+"ranksIMPlog0.csv", metric)
                models = list(set((clean_dataset["Model"].to_list()+clean_dataset["Gt"].to_list())))
                models.sort()
               
                model_typ_step_data = model_typology_robustness_data(models, typology_noising_step)
                for typology_step in range(typology_noising_step, 100 + typology_noising_step,
                                            typology_noising_step):
                    root=output_noisedresult_folder+"IMPlog0"
                    classification_files = select_fixed_magnitude_paths(root, noise_magnitude)
                    fixed_metric_files = filter_fixed_typology(classification_files, typology, typology_step)
                    for file in fixed_metric_files:
                        dataset = read_classification_files(f"{root}/{file}", metric)
                        if len(dataset)<=0: continue
                        for model in models:
                            model_step_data = []
                            records1 = dataset[dataset['Model'] == model]
                            records2 = dataset[dataset['Gt'] == model]
                            records = pd.concat([records1,records2])
                            for indice, riga in records.iterrows():
                                if metric == 'mse':
                                        noised_metric = riga["mse_n"]
                                        clean_metric_record = clean_dataset[
                                            (clean_dataset['Model'] == riga["Model"]) & (
                                                    clean_dataset['Gt'] == riga["Gt"])]
                                        clean_metric = clean_metric_record["mse_n"].values[0]
                                        robustness = 1 - abs(noised_metric - clean_metric)
                                        model_step_data.append(robustness)
                                elif metric == 'mae':
                                        noised_metric = riga["mae_n"]
                                        clean_metric_record = clean_dataset[
                                            (clean_dataset['Model'] == riga["Model"]) & (
                                                    clean_dataset['Gt'] == riga["Gt"])]
                                        clean_metric = clean_metric_record["mae_n"].values[0]
                                        robustness = 1 - abs(noised_metric - clean_metric)
                                        model_step_data.append(robustness)
                                elif metric =='median':
                                        noised_metric = riga["median_err_n"]
                                        clean_metric_record = clean_dataset[
                                            (clean_dataset['Model'] == riga["Model"]) & (
                                                    clean_dataset['Gt'] == riga["Gt"])]
                                        clean_metric = clean_metric_record["median_err_n"].values[0]
                                        robustness = 1 - abs(noised_metric - clean_metric)
                                        model_step_data.append(robustness)
                            model_typ_step_data.add_robustness((model, typology_step), model_step_data)
                fig = plt.figure(figsize=(10, 5))
                plt.rcParams.update({'font.size': 10})
                xticks = []
                xtick_names = []
                x_position_increment = 1 / len(models)
                x = 1
                for typology_step in range(typology_noising_step, 100 + typology_noising_step,
                                            typology_noising_step):
                    xticks.append(x)
                    x = x + 2
                    xtick_names.append(typology_step)

                for model_idx in range(len(models)):
                    values = []
                    positions = []
                    position = 1
                    for typology_step in range(typology_noising_step, 100 + typology_noising_step,
                                                typology_noising_step):
                        for x in range(len(model_typ_step_data.models_magnitude)):
                            if model_typ_step_data.models_magnitude[x][1] == typology_step and \
                                    model_typ_step_data.models_magnitude[x][0] == models[model_idx]:
                                values.append(model_typ_step_data.robustness[x])
                        positions.append(position + x_position_increment * model_idx)
                        position += 2

                    plt.boxplot(values, positions=positions, notch=False, patch_artist=True,
                                boxprops=dict(facecolor=colors_models[model_idx], color="#000000"),
                                capprops=dict(color="#000000"),
                                whiskerprops=dict(color="#000000"),
                                flierprops=dict(color="#000000",
                                                markeredgecolor="#000000"),
                                widths=(x_position_increment / 3) * 2
                                )
                plt.xticks(xticks, xtick_names)
                labels_legend =['M1', 'M2', 'M3', 'M4', 'ETR']
                legend_lines = [plt.Line2D([0], [0], color=colors_models[0], lw=2),
                                plt.Line2D([0], [0], color=colors_models[1], lw=2),
                                plt.Line2D([0], [0], color=colors_models[2], lw=2),
                                plt.Line2D([0], [0], color=colors_models[3], lw=2),
                                plt.Line2D([0], [0], color=colors_models[4], lw=2)]
                plt.legend(legend_lines, labels_legend)
                    
                plt.xlabel("% noise")
                plt.ylabel("Robustness")
                if not os.path.exists(f"./plot/robustness/Noise{noise_magnitude}%"):
                    os.makedirs(f"./plot/robustness/Noise{noise_magnitude}%")
                plt.savefig(
                    f"./plot/robustness/Noise{noise_magnitude}%/relative_{metric}_{typology}.png", bbox_inches='tight')

                plt.ylim(0, 1.1)
                if not os.path.exists(f"./plot/robustness/Noise{noise_magnitude}%"):
                    os.makedirs(f"./plot/robustness/Noise{noise_magnitude}%")
                plt.savefig(
                    f"./plot/robustness/Noise{noise_magnitude}%/absolute_{metric}_{typology}.png", bbox_inches='tight')
                plt.close(fig)

class graph_two_data:
    def __init__(self, models_data):
        self.models = []
        for model in models_data:
            self.models.append(model)
        self.robustness = []
        for model in models_data:
            self.robustness.append([])

    def add_robustness(self, model, robustness_data):
        index_model = self.models.index(model)
        self.robustness[index_model] += robustness_data

def select_paths_noised_pure(root, typology):
    files = os.listdir(root)
    if "D0_P0_R0_I0_U0_P0_[V0,N0,M0(0)]" in files:
        files.remove("D0_P0_R0_I0_U0_P0_[V0,N0,M0(0)]")
    if typology == "null_values":
        return list(filter(lambda x: "V100" in x, files))
    elif typology == "incorrect":
        return list(filter(lambda x: "N100" in x, files))
    elif typology == "imprecise":
        return list(filter(lambda x: "M100" in x, files))

def plot_robustness_by_noise():
    metrics=["mae", "mse", "median"]

    possible_noise_magnitude = []
    for noise_magnitude in range(noise_magnitude_step, 51, noise_magnitude_step):
        possible_noise_magnitude.append(noise_magnitude)
    
    for typology in ["null_values", "incorrect", "imprecise"]:
        clean_dataset = read_clean_classification_files(output_cleanfolder+"ranksIMPlog0.csv", "all")
        models = list(set((clean_dataset["Model"].to_list()+clean_dataset["Gt"].to_list())))
        models.sort()

        robustness_data = graph_two_data(models)
        for metric in metrics:
            clean_dataset = read_clean_classification_files(output_cleanfolder+"ranksIMPlog0.csv", metric)
            root=output_noisedresult_folder+"IMPlog0"
            noised_files = select_paths_noised_pure(root, typology)
            for file in noised_files:
                dataset = read_classification_files(f"{root}/{file}", metric)
                if len(dataset)<=0: continue
                for model in models:
                    model_data = []
                    records1 = dataset[dataset['Model'] == model]
                    records2 = dataset[dataset['Gt'] == model]
                    records = pd.concat([records1,records2])
                    for indice, riga in records.iterrows():
                        if metric == 'mse':
                            noised_metric = riga["mse_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["mse_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
                        elif metric == 'mae':
                            noised_metric = riga["mae_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["mae_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
                        elif metric == 'median':
                            noised_metric = riga["median_err_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["median_err_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
                    robustness_data.add_robustness(model, model_data)
        values = []
        labels = []
        labels_legend =['M1', 'M2', 'M3', 'M4', 'ETR']
        for x in range(len(robustness_data.models)):
            values.append(robustness_data.robustness[x])
            labels.append(labels_legend[x])

        # legend_lines = [plt.Line2D([0], [0], color=colors_models[0], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[1], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[2], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[3], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[4], lw=2)]
        # plt.legend(legend_lines, labels_legend)

        fig = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        boxplot = plt.boxplot(values,
                                vert=True,
                                patch_artist=True,
                                labels=labels)
        for patch, color in zip(boxplot['boxes'], colors_models):
            patch.set_facecolor(color)
        
        plt.xlabel("Models")
        plt.ylabel("Robustness")
        
        if not os.path.exists(f"./plot/robustness_pure"):
            os.makedirs(f"./plot/robustness_pure")
        plt.savefig(f"./plot/robustness_pure/relative_Noise_{typology}.png", bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        boxplot = plt.boxplot(values,
                                vert=True,
                                patch_artist=True,
                                labels=labels)
        for patch, color in zip(boxplot['boxes'], colors_models):
            patch.set_facecolor(color)
        plt.xlabel("Models")
        plt.ylabel("Robustness")
        plt.ylim(0, 1.1)
        if not os.path.exists(f"./plot/robustness_pure"):
            os.makedirs(f"./plot/robustness_pure")
        plt.savefig(f"./plot/robustness_pure/absolute_Noise_{typology}.png", bbox_inches='tight')
        plt.close(fig)

class feature_robustness_data:
    def __init__(self, features):
        self.features = []
        for feature in features:
            self.features.append(feature)
        self.robustness = []
        for feature in features:
            self.robustness.append([])

    def add_robustness(self, feature, robustness_data):
        index_model = self.features.index(feature)
        self.robustness[index_model] += robustness_data

def rename_feature(str_feature):
    if str_feature=="duration_phase": return "duration"
    elif str_feature=="preproc_priority": return "priority"
    elif str_feature=="reassignment_count": return "num. employees"
    elif str_feature=="preproc_impact": return "impact"
    elif str_feature=="preproc_urgency": return "urgency"
    else: return ""

def select_paths_feature(root, feature):
    files = os.listdir(root)
    if feature == "duration_phase":
        return list(filter(lambda x: f"D1" in x, files))
    elif feature == "preproc_priority":
        return list(filter(lambda x: f"P1_R" in x, files))
    elif feature == "reassignment_count":
        return list(filter(lambda x: f"R1" in x, files))
    elif feature == "preproc_impact":
        return list(filter(lambda x: f"I1" in x, files))
    elif feature == "preproc_urgency":
        return list(filter(lambda x: f"U1" in x, files))

def select_paths_feature_pure(root, feature):
    files = os.listdir(root)
    if feature == "duration_phase":
        return list(filter(lambda x: f"D1_P0_R0_I0_U0" in x, files))
    elif feature == "preproc_priority":
        return list(filter(lambda x: f"D0_P1_R0_I0_U0" in x, files))
    elif feature == "reassignment_count":
        return list(filter(lambda x: f"D0_P0_R1_I0_U0" in x, files))
    elif feature == "preproc_impact":
        return list(filter(lambda x: f"D0_P0_R0_I1_U0" in x, files))
    elif feature == "preproc_urgency":
        return list(filter(lambda x: f"D0_P0_R0_I0_U1" in x, files))

def plot_box_robustness_features():
    metrics=["mae", "mse", "median"]

    possible_noise_magnitude = []
    for noise_magnitude in range(noise_magnitude_step, 51, noise_magnitude_step):
        possible_noise_magnitude.append(noise_magnitude)
    
    clean_dataset = read_clean_classification_files(output_cleanfolder+"ranksIMPlog0.csv", "all")
    models = list(set((clean_dataset["Model"].to_list()+clean_dataset["Gt"].to_list())))
    models.sort()
    for model in models:
        robustness_data = feature_robustness_data(features)
        for feature in features:
            root=output_noisedresult_folder+"IMPlog0"
            
            experiments = select_paths_feature(root, feature)
            model_data = []
            for experiment in experiments:
                for metric in metrics:
                    clean_dataset = read_clean_classification_files(output_cleanfolder+"ranksIMPlog0.csv", metric)
                    dataset = read_classification_files(f"{root}/{experiment}", metric)
                    if len(dataset)<=0: continue
                    records1 = dataset[dataset['Model'] == model]
                    records2 = dataset[dataset['Gt'] == model]
                    records = pd.concat([records1,records2])

                    for indice, riga in records.iterrows():
                        if metric == 'mse':
                            noised_metric = riga["mse_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["mse_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
                        elif metric == 'mae':
                            noised_metric = riga["mae_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["mae_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
                        elif metric == 'median':
                            noised_metric = riga["median_err_n"]
                            clean_metric_record = clean_dataset[
                                (clean_dataset['Model'] == riga["Model"]) & (
                                        clean_dataset['Gt'] == riga["Gt"])]
                            clean_metric = clean_metric_record["median_err_n"].values[0]
                            robustness = 1 - abs(noised_metric - clean_metric)
                            model_data.append(robustness)
            robustness_data.add_robustness(feature, model_data)
        values = []
        labels = []
        for x in range(len(robustness_data.features)):
            values.append(robustness_data.robustness[x])
            labels.append(rename_feature(robustness_data.features[x]))

        # legend_lines = [plt.Line2D([0], [0], color=colors_models[0], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[1], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[2], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[3], lw=2),
        #                 plt.Line2D([0], [0], color=colors_models[4], lw=2)]

        fig = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        boxplot = plt.boxplot(values,
                                vert=True,
                                patch_artist=True,
                                labels=labels)
        for patch, color in zip(boxplot['boxes'], colors_models):
            patch.set_facecolor(color)
        # plt.legend(legend_lines, labels)
        plt.xlabel("Features")
        plt.ylabel("Robustness")
        if not os.path.exists(f"./plot/robustness_features"):
            os.makedirs(f"./plot/robustness_features")
        plt.savefig(f"./plot/robustness_features/relative_{model}.png", bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10, 5))
        plt.rcParams.update({'font.size': 10})
        boxplot = plt.boxplot(values,
                                vert=True,
                                patch_artist=True,
                                labels=labels)
        for patch, color in zip(boxplot['boxes'], colors_models):
            patch.set_facecolor(color)
        # plt.legend(legend_lines, labels)
        plt.xlabel("Features")
        plt.ylabel("Robustness")
        plt.ylim(0, 1.1)
        if not os.path.exists(f"./plot/robustness_features"):
            os.makedirs(f"./plot/robustness_features")
        plt.savefig(f"./plot/robustness_features/absolute_{model}.png", bbox_inches='tight')
        plt.close(fig)

def main_analysis():
    plot_costs()
    # plot_performance()

    # plot_single_err("mae")
    # plot_single_err("mse")
    # plot_single_err("median")

    # plot_robustness()
    # plot_robustness_by_noise()
    # plot_box_robustness_features()