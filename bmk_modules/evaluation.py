import pandas as pd
import numpy as np
import sklearn.metrics as metrics

## TODO: NORMALIZZAZIONE TOGLIE DEI VALORI
def normalize_df_col(df, cols):
    for c in cols:
        mdev = df[c].median()
        df = df[np.abs(df[c]-mdev)/mdev <= 2]
        df[c] = (df[c]-df[c].min()) / (df[c].max()-df[c].min())
    return df

def compare_models(enriched_log, case_k, outfilename):
    results=[]
    cols = [col for col in enriched_log.columns if 'gt' in col]

    for gt_k in cols:
        df_normal_model = normalize_df_col(enriched_log, [gt_k])

        for gt_h in cols:
            if gt_k != gt_h:
                param_dict={}
                
                df_normal_gt = normalize_df_col(enriched_log, [gt_h])
                if len(df_normal_model) != len(df_normal_gt): continue
                
                y = df_normal_model[gt_k].to_list()
                x = df_normal_gt[gt_h].to_list()
                mse = metrics.mean_squared_error(y, x)
                mae = metrics.mean_absolute_error(y, x)
                mad = metrics.median_absolute_error(y, x)
                param_dict["mse"] = mse
                param_dict["mae"] = mae
                param_dict["mad"] = mad
                param_dict["model-gt_id"] = gt_k+"-"+gt_h
                results.append(param_dict)
    
    resultsDf = pd.DataFrame(results)
    resultsDf.to_csv(outfilename, index=False)
    return resultsDf










    df_log = log_by_case

    results = []
    with open(parameter_file) as csv_r:
        reader = csv.DictReader(csv_r)
        for param_dict in reader:
            df_model = compute_cost(log_by_case, case_k, param_dict)
            
            # df_log['incident_id'] = df_log['incident_id'].astype(str)
            # df_model['incident_id'] = df_model['incident_id'].astype(str)
            # df = pd.merge(df_log[[case_k, cost_k]], df_model, how='inner', on=[case_k])

            # df = normalize_df_col(df, [cost_k, "cost_model"])

            if len(df)>0:
                y = df[cost_k].to_list()
                x = df["cost_model"].to_list()
                mse = metrics.mean_squared_error(y, x)
                mae = metrics.mean_absolute_error(y, x)
                mad = metrics.median_absolute_error(y, x)
                param_dict["mse"] = mse
                param_dict["mae"] = mae
                param_dict["mad"] = mad
                results.append(param_dict)

    resultsDf = pd.DataFrame(results)
    # resultsDf.to_csv("./results.csv")
    with open(parameter_file, 'w', newline='') as param_obj:            
        dict_writer = csv.DictWriter(param_obj, results[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(results)
    return resultsDf