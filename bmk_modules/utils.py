import csv
import numpy as np
import pandas as pd

def format_dataset_by_incidents(logfile, case_k):
    
    original_log = pd.read_csv(logfile)

    incidents = original_log[case_k].unique()
    process_durations = []
    reassignment_counts = []
    for incident in incidents:
        incident_records = original_log[original_log[case_k] == incident]
        durations = incident_records["duration_phase"]
        original_reassignment_counts = incident_records["reassignment_count"]
        total_duration = 0
        for duration in durations:
            if not np.isnan(duration):
                total_duration += duration
        process_durations.append(total_duration)
        reassignment_count = max(original_reassignment_counts)
        reassignment_counts.append(reassignment_count)

    df_log = original_log.copy()
    df_log = df_log.drop_duplicates(subset=case_k, keep="last")
    df_log.drop(columns=["duration_phase"], inplace=True)
    df_log["duration_process"] = process_durations
    df_log["reassignment_count"] = reassignment_counts

    # df_merged = pd.merge(df_log, df_align, how='inner', on=[case_k])
    return df_log


def detect_process_elements(log_by_case):
    activities = []
    deviations = []
    for index, row in log_by_case.iterrows():
        all = row['trace'].split(";")[:-1]
        for i in range(0, len(all)):
            elem = all[i]
            if elem[0] == "m" or elem[0] == "s" or elem[0] == "r":
                deviations.append(elem)
            else:
                activities.append(elem)

    activities = list(dict.fromkeys(activities))
    deviations = list(dict.fromkeys(deviations))
    return activities, deviations


def write_models_parameters(models_list, parameter_file):
    with open(parameter_file, 'w', newline='') as param_obj:
        dict_writer = csv.DictWriter(param_obj, models_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(models_list)


def filename_generation(features, percentage, combination, magnitude):
    duration = 1 if "duration_phase" in features else 0
    priority = 1 if "preproc_priority" in features else 0
    reassignment = 1 if "reassignment_count" in features else 0
    impact = 1 if "preproc_impact" in features else 0
    urgency = 1 if "preproc_urgency" in features else 0
    features_part = f"D{duration}_P{priority}_R{reassignment}_I{impact}_U{urgency}"
    file_name = f"{features_part}_P{percentage}_[V{combination[0]},N{combination[1]},M{combination[2]}({magnitude})]"
    return file_name