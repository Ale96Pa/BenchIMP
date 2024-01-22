def filename_generation(features, percentage, combination, magnitude):
    duration = 1 if "duration_phase" in features else 0
    priority = 1 if "preproc_priority" in features else 0
    reassignment = 1 if "reassignment_count" in features else 0
    impact = 1 if "preproc_impact" in features else 0
    urgency = 1 if "preproc_urgency" in features else 0
    features_part = f"D{duration}_P{priority}_R{reassignment}_I{impact}_U{urgency}"
    file_name = f"{features_part}_P{percentage}_[V{combination[0]},N{combination[1]},M{combination[2]}({magnitude})]"
    return file_name