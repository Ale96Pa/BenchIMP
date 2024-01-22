import numpy as np
import pandas as pd


"""
Ground Truth 1 (GT1)
Cost of non-compliance of each trace according to [Kieninger]
[Kieninger] Axel Kieninger, Florian Berghoff, Hansj ̈org Fromm, and Gerhard Satzger.
Simulation-Based Quantification of Business Impacts Caused by Service Incidents.
"""
def build_gt1(log, log_by_case, case_k, fract=1):
    df_original = log
    df_log_case = log_by_case[[case_k, "trace"]]
    grouped_by_case = df_original.groupby([case_k])
    case_ids = set(df_original[case_k].to_list())
    l_dict = []

    for caseID in case_ids:
        single_case_df = grouped_by_case.get_group(caseID)
        trace_id = single_case_df[case_k].to_list()[0]

        trace = df_log_case.query(case_k + " == '" + str(trace_id) + "'")["trace"].to_string()
        activities = trace.split(";")
        durations = single_case_df['duration_phase'].to_list()

        avg_duration = sum(durations) / len(durations)

        cost = 0
        counter_miss = 0
        for i in range(0, len(activities)):
            if "s" in activities[i]:
                cost += avg_duration
                counter_miss += 1
            elif "r" in activities[i] or "m" in activities[i]:
                cost += round(durations[i - counter_miss - 1], 2)
        l_dict.append({case_k: trace_id, "gt2_" + str(fract): abs(cost)})
    return pd.DataFrame(l_dict)


"""
Ground Truth 2 (GT2)
Cost of traces for SLA violations according to [Moura]
[Moura] MOURA, Antão, et al. A quantitative approach to IT investment allocation to 
improve business results. In: Seventh IEEE International Workshop on Policies 
for Distributed Systems and Networks (POLICY'06). IEEE, 2006. p. 9 pp.-95.
"""


def build_gt2(log, log_by_case, case_k, fract=1):
    df_original = log
    df_log_case = log_by_case[[case_k, "trace"]]
    grouped_by_case = df_original.groupby([case_k])
    case_ids = set(df_original[case_k].to_list())
    l_dict = []

    for caseID in case_ids:
        single_case_df = grouped_by_case.get_group(caseID)
        trace_id = single_case_df[case_k].to_list()[0]

        trace = df_log_case.query(case_k + " == '" + str(trace_id) + "'")["trace"].to_string()
        activities = trace.split(";")
        durations = single_case_df['duration_phase'].to_list()

        activities = [item for item in activities if "s" not in item]
        prio_val = np.nanmax(log_by_case["preproc_priority"].to_list())

        cost = 0
        for i in range(0, len(activities)):
            if len(activities[i]) > 1 and i > 0:
                cost = round(durations[i - 1] * prio_val, 2)
        l_dict.append({case_k: trace_id, "gt3_" + str(fract): abs(cost)})

    return pd.DataFrame(l_dict)


"""
Ground Truth 3 (GT3)
Cost of the trace based on person-hours metric according to [Dumas]
[Dumas] Marlon Dumas, Marcello La Rosa, Jan Mendling, Hajo A Reijers, et al.
Fundamentals of business process management, volume 1. Springer, 2013.
"""
def build_gt3(log_by_case, case_k, fract):
    l_dict = []
    for index, row in log_by_case.iterrows():
        duration_process = row['duration_process']
        num_employee = row['reassignment_count'] + 1
        cost = round(duration_process * num_employee)
        l_dict.append({case_k: row[case_k], "gt4_" + str(fract): cost * fract})
    return pd.DataFrame(l_dict)


"""
Ground Truth 4 (GT4)
Cost of the traces according to the model defined in [4]
[4] Sasha Romanosky. Examining the costs and causes of cyber incidents.
Journal of Cybersecurity, page tyw001, August 2016.
"""
def build_gt4(log_by_case, case_k, fract):
    l_dict = []
    for index, row in log_by_case.iterrows():
        num_employee = row['reassignment_count'] + 1
        impact = row['preproc_impact']
        cost = 2.67 * int(num_employee) * impact
        l_dict.append({case_k: row[case_k], "gt5_" + str(fract): cost * fract})
    return pd.DataFrame(l_dict)


def ground_thruths_main(log, log_by_case, test, root_path):
    alpha = conf.alpha
    full_output_path = conf.bmk_output
    gt1 = build_gt1(log_by_case, 'incident_id')
    gt2 = build_gt2(log, log_by_case, 'incident_id')
    df_gts = pd.merge(gt1, gt2, how='inner', on=['incident_id'])
    gt3 = build_gt3(log, log_by_case, 'incident_id')
    df_gts = pd.merge(df_gts, gt3, how='inner', on=['incident_id'])

    for fract in alpha:
        gt4 = build_gt4(log_by_case, 'incident_id', fract)
        gt5 = build_gt5(log_by_case, 'incident_id', fract)
        df_gts = pd.merge(df_gts, gt4, how='inner', on=['incident_id'])
        df_gts = pd.merge(df_gts, gt5, how='inner', on=['incident_id'])

    gtfull_fullpath = f"{root_path}{full_output_path}"
    if not os.path.exists(gtfull_fullpath):
        os.makedirs(gtfull_fullpath)

    if test != "":
        filename_gt_full = f"GtFull_{test}.csv"
    else:
        filename_gt_full = f"GtFull.csv"

    df_full = pd.merge(log_by_case, df_gts, how='inner', on=['incident_id'])
    df_full.to_csv(f"{gtfull_fullpath}/{filename_gt_full}", index=False)
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