import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tabular_augmentation import mixup_augmentation_with_weight
from bmk_modules.utils import format_dataset_by_incidents

method="vanilla"
seed=42
last_id=10000001

def augment_log(logpath, outpath, features_training, feature_target):
    dfevent = pd.read_csv(logpath)
    df = format_dataset_by_incidents(logpath, "incident_id")
    
    df = df[features_training+[feature_target]]

    x_train, x_test, y_train, y_test = train_test_split(df[features_training], 
                      df[feature_target], test_size=0.3, random_state=seed)
    x_few_train, _, y_few_train, _ = train_test_split(x_train, y_train, 
                        train_size=150, random_state=seed)
    
    x_synthesis, y_synthesis, sample_weight = mixup_augmentation_with_weight(
            x_few_train, y_few_train, oversample_num=200, alpha=1, beta=1, 
            mixup_type=method, seed=seed, rebalanced_ita=1)
    
    dicts=[]
    for i in range(0,len(x_synthesis)):
        dict_augmentation={}
        dict_augmentation["incident_id"] = "INC"+str(last_id+i)
        dict_augmentation[feature_target] = y_synthesis[i]
        for j in range(0,len(features_training)):
            feat = features_training[j]
            dict_augmentation[feat] = x_synthesis[i][j]
        dicts.append(dict_augmentation)

    all_ids = set(dfevent["incident_id"].to_list())
    case_ids = random.sample(all_ids, min(len(dicts),len(all_ids)))
    
    i=0
    full_df = []
    grouped_by_case = dfevent.groupby(["incident_id"])
    for caseID in case_ids:
        augmented_dict = dicts[i]
        single_case_df = grouped_by_case.get_group(caseID)
        for k_dict in augmented_dict.keys():
            single_case_df[k_dict] = augmented_dict[k_dict]
        full_df.append(single_case_df)    
        i+=1
    
    full_augmented_log = pd.concat(full_df, ignore_index=True)
    full_augmented_log.to_csv(outpath, index=False)
    return full_augmented_log