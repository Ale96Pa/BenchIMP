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

    df_full = pd.merge(log_by_case, df_gts, how='inner', on=['incident_id'])
    if eventlog!="": 
        outfilename = eventlog.split("/")
        df_full.to_csv(outfolder+outfilename[len(outfilename)-1], index=False)
    else:
        folderparams = outfolder.split("/")
        df_full.to_csv(outfolder+folderparams[len(folderparams)-2]+".csv", index=False)
    return df_full