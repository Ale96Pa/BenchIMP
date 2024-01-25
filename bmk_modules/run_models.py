import pandas as pd
from input.models import Models

def cost_computation(eventlog, log_by_case, outfolder):
    
    log=None
    if eventlog!="": log = pd.read_csv(eventlog)

    gt1 = Models.gt1(log, log_by_case, 'incident_id')
    gt2 = Models.gt2(log, log_by_case, 'incident_id')
    gt3 = Models.gt3(log_by_case, 'incident_id')
    gt4 = Models.gt4(log_by_case, 'incident_id')
    
    df_gts = pd.merge(gt1, gt2, how='inner', on=['incident_id'])
    df_gts = pd.merge(df_gts, gt3, how='inner', on=['incident_id'])
    df_gts = pd.merge(df_gts, gt4, how='inner', on=['incident_id'])

    # gtfull_fullpath = f"{root_path}{full_output_path}"
    # if not os.path.exists(gtfull_fullpath):
    #     os.makedirs(gtfull_fullpath)

    # if test != "":
    #     filename_gt_full = f"GtFull_{test}.csv"
    # else:
    #     filename_gt_full = f"GtFull.csv"

    df_full = pd.merge(log_by_case, df_gts, how='inner', on=['incident_id'])
    if eventlog!="": 
        outfilename = eventlog.split("/")
        df_full.to_csv(outfolder+outfilename[len(outfilename)-1], index=False)
    else:
        folderparams = outfolder.split("/")
        df_full.to_csv(outfolder+folderparams[len(folderparams)-2]+".csv", index=False)
    return df_full


# def ground_thruths_clean_gt(noised_log_by_case, clean_gt_dfFull, test, root_path):
#     full_output_path = conf.bmk_output
#     gtfull_fullpath = f"{root_path}{full_output_path}"
#     if not os.path.exists(gtfull_fullpath):
#         os.makedirs(gtfull_fullpath)

#     if test != "":
#         filename_gt_full = f"GtFull_{test}.csv"
#     else:
#         filename_gt_full = f"GtFull.csv"

#     gt_headers = ['incident_id']
#     for header in clean_gt_dfFull.columns.values:
#         if "gt" in header.lower():
#             gt_headers.append(header)
#     df_gts = clean_gt_dfFull[gt_headers].copy()
#     df_full = pd.merge(noised_log_by_case, df_gts, how='inner', on=['incident_id'])
#     df_full.to_csv(f"{gtfull_fullpath}/{filename_gt_full}", index=False)
#     return df_full