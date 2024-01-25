import logging, random, os
from pebble import ProcessPool

from bmk_modules.utils import format_dataset_by_incidents, filename_generation
import config

ratio_step = config.ratio_step
magnitude_step = config.magnitude_step
messing_step = config.messing_step
features_list = config.features_list
features_type = config.features_type

logging_folder = config.logging_folder
tmp_folder=config.tmp_folder
output_noisedLogs = config.output_noisedlog_folder
output_logByCase = config.output_noisedlogbycase_folder

in_memory = config.in_memory

"""
Select the indexes of the row to noise in such a way that a single row is not 
"noised" with more than one noise type
@Input:
    dataset_rows: number of rows of the input dataset
    errors_number: number of elements to be noised 
    dirt_entries: array of indexes of the previous "noised" rows
@Output:
    indexes: an array of indexes of not "noised" row
"""
def extract_indexes(dataset, errors_number, dirt_entries):
    possible_indexes = dataset.reset_index(drop=True).index.values.tolist()
    clean_indexes = possible_indexes
    extracted_indexes = []
    for index in dirt_entries:
        clean_indexes.remove(index)
    if len(clean_indexes) >= errors_number:
        for x in range(0, errors_number):
            index = random.choice(clean_indexes)
            dirt_entries.append(index)
            extracted_indexes.append(index)
            clean_indexes.remove(index)
    return extracted_indexes


"""
Extract an array of indexes representing clean rows. Substitute the original 
values of considered features for the extracted rows with Null Values.
@Input:
    error_number: number of rows that must be noised
    dataset: a pointer to the dirt dataset, python works by object reference, it is used also ad output
    dirt_entries: array of indexes of previously noised rows
    features: array of features use in noising strategies
@Output: None
"""
def null_values_generator(errors_number, noised, dirt_entries, features):
    indexes = extract_indexes(noised, errors_number, dirt_entries)
    for row_index in indexes:
        for feature in features:
            noised.loc[row_index, feature] = None


"""
Generate incorrect values, basing the new value decision on the type of the input features.
@Input:
    errors_number: number of errors to introduce into the dataset
    original: a pointer to the clean dataset
    noised: a pointer to noised dataset, it represent also the output of the function
    dirt_entries = indexes of the entries previously noised
    features = the list of feature which must be noised
@Output: None
"""
def incorrect_value_generator(errors_number, original, noised, dirt_entries, features):
    indexes = extract_indexes(original, errors_number, dirt_entries)
    for row_index in indexes:
        for feature in features:
            index_feature = features_list.index(feature)
            feature_type = features_type[index_feature]
            if feature not in ["preproc_priority", "preproc_impact", "preproc_urgency"]:
                match feature_type:
                    case "integer":
                        if feature == 'reassignment_count':
                            dom_min: int = min(original[[feature]].values)[0]
                            dom_max: int = max(original[[feature]].values)[0]
                            dom_extension = dom_max - dom_min
                            magnitude = random.uniform(50, 60)
                            deviation = (dom_extension * magnitude) / 100
                            if deviation < 1:
                                deviation = 1
                            new_value = round(dom_max + deviation)
                            noised.loc[row_index, feature] = new_value
                        else:
                            dom_min: int = min(original[[feature]].values)[0]
                            dom_max: int = max(original[[feature]].values)[0]
                            dom_extension = dom_max - dom_min
                            magnitude = random.uniform(50, 60)
                            deviation = (dom_extension * magnitude) / 100
                            if deviation < 1:
                                deviation = 1
                            grater_flag = random.choice([True, False])
                            if grater_flag:
                                new_value = round(dom_max + deviation)
                            else:
                                new_value = round(dom_min - deviation)
                                if new_value < 0:
                                    new_value = 0
                            noised.loc[row_index, feature] = new_value
                    case "float":
                        dom_min = min(original[[feature]].values)
                        dom_max = max(original[[feature]].values)
                        dom_extension = dom_max - dom_min
                        magnitude = random.uniform(50, 60)
                        deviation = (dom_extension * magnitude) / 100
                        grater_flag = random.choice([True, False])
                        if grater_flag:
                            new_value = dom_max + deviation
                        else:
                            new_value = dom_min - deviation
                            if new_value < 0:
                                new_value = 0
                        noised.loc[row_index, feature] = new_value
                    case _:
                        change_valid = False
                        while not change_valid:
                            new_value = noised.loc[row_index][feature]
                            new_value = f"{new_value}*"
                            column_values = original[[feature]].values
                            if not column_values.__contains__(new_value):
                                noised.loc[row_index, feature] = new_value
                                change_valid = True
            else:
                values = extract_domain_values(feature, original)
                dom_min: int = min(values)
                dom_max: int = max(values)
                minor_flag = random.choice([True, False])
                if minor_flag:
                    new_value = dom_min - 1
                else:
                    new_value = dom_max + 1
                noised.loc[row_index, feature] = new_value


"""
Extract the values assumed by a given feature inside a given dataset
@Input:
    features: the feature to analyse
    dataset: the dataset on which compute the feature domain
@Output: values: the value assumed by the feature into the dataset
"""
def extract_domain_values(feature, dataset):
    values = []
    dataset_values = dataset[feature].values.tolist()
    for value in dataset_values:
        if value not in values:
            values.append(value)
    return values


"""
Generate incorrect values, basing the new value decision on the type of the input features.
@Input:
    errors_number: number of errors to introduce into the dataset
    original: a pointer to the clean dataset
    noised: a pointer to noised dataset, it represent also the output of the function
    dirt_entries = indexes of the entries previously noised
    features = the list of feature which must be noised
@Output: None
"""
def inaccurate_value_generator(original, noised, features, magnitude, indexes):
    for row_index in indexes:
        for feature in features:
            feature_index = features_list.index(feature)
            match features_type[feature_index]:
                case "integer":
                    dom_min = int(min(original[[feature]].values)[0])
                    dom_max = int(max(original[[feature]].values)[0])
                    if dom_min <= noised.iloc[row_index][feature] <= dom_max:
                        if original.iloc[row_index][feature] == dom_min:
                            increase_flag = True
                        elif original.iloc[row_index][feature] == dom_max:
                            increase_flag = False
                        else:
                            increase_flag = random.choice([True, False])
                        extracted_magnitude = random.uniform(0.01, magnitude)
                        deviation = round((noised.iloc[row_index][feature] * extracted_magnitude) / 100)
                        if deviation < 1:
                            deviation = 1
                        if increase_flag:
                            new_value = round(noised.iloc[row_index][feature] + deviation)
                            if new_value > dom_max:
                                new_value = dom_max
                            noised.loc[row_index, feature] = round(new_value)
                        else:
                            new_value = round(noised.iloc[row_index][feature] - deviation)
                            if new_value < dom_min:
                                new_value = dom_min
                            noised.loc[row_index, feature] = round(new_value)
                case "float":
                    dom_min = min(original[[feature]].values)
                    dom_max = max(original[[feature]].values)
                    if dom_min <= noised.iloc[row_index][feature] <= dom_max:
                        extracted_magnitude = random.uniform(0.01, magnitude)
                        deviation = (noised.iloc[row_index][feature] * extracted_magnitude) / 100
                        if original.iloc[row_index][feature] == dom_min:
                            increase_flag = True
                        elif original.iloc[row_index][feature] == dom_max:
                            increase_flag = False
                        else:
                            increase_flag = random.choice([True, False])
                        if increase_flag:
                            new_value = noised.iloc[row_index][feature] + deviation
                            if new_value > dom_max:
                                new_value = dom_max
                            noised.loc[row_index, feature] = new_value
                        else:
                            new_value = noised.iloc[row_index][feature] - deviation
                            if new_value < dom_min:
                                new_value = dom_min
                            noised.loc[row_index, feature] = new_value
                case _:
                    feature_domain = extract_domain_values(feature, original)
                    changed_value = False
                    while not changed_value:
                        index = random.randint(0, len(feature_domain) - 1)
                        if len(feature_domain) > 1:
                            new_value = feature_domain[index]
                            if noised.iloc[row_index][feature] != new_value:
                                noised.loc[row_index, feature] = new_value
                                changed_value = True
                        else:
                            changed_value = True


"""
Save the output dataset in a .csv file punted by the output_path configuration configuration parameter.
If the output_path do not exist, create it and than save. The method save also a .txt file that contain the
indexes of the noised rows.
@Input:
    features: list of features noised
    mess_percentage: the percentage of noise introduced into the log file, w.r.t. the number of noised rows over the
        total number of rows
    combination: the combination of noise types percentages
    magnitude: the magnitude of noise in the imprecise values noise generation
    dataset: the output dataset
    noised_entries: the list of noised entries
@Output: None
"""
def save_output(features, mess_percentage, combination, magnitude, dataset, 
                flag_logs, noised_entries, methodology_one, methodology_two, logID):
    if flag_logs is False:
        output_path = output_logByCase+logID+"/"
    else:
        output_path = output_noisedLogs+logID+"/"
    file_name = f"{filename_generation(features, mess_percentage, combination, magnitude)}"
    
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    dataset = dataset[dataset['incident_id'].notna()]
    dataset.to_csv(f"{output_path}/{file_name}.csv", index=False)
    noised = f"{output_path}/{file_name}.txt"
    with open(noised, 'w') as f:
        counter = 0
        for entry in noised_entries:
            if counter < methodology_one:
                f.write(f"V: {entry}\n")
            elif counter < methodology_one + methodology_two:
                f.write(f"N: {entry}\n")
            else:
                f.write(f"M: {entry}\n")
            counter += 1


def noising_orchestrator(original_dataset, mess_percentage, combination_ratio, features, features_filename, in_memory: bool,
                 flag_logs, logID):
    results = []
    dirty_dataset = original_dataset.copy(deep=True)
    number_entries = round((mess_percentage * dirty_dataset.shape[0]) / 100)
    noised_entries = []
    if combination_ratio[0] != 0:
        methodology_one = round((combination_ratio[0] * number_entries) / 100)
        if not methodology_one < 1:
            null_values_generator(methodology_one, dirty_dataset, noised_entries, features)
    else:
        methodology_one = 0
    if combination_ratio[1] != 0:
        methodology_two = round((combination_ratio[1] * number_entries) / 100)
        if not methodology_two < 1:
            incorrect_value_generator(methodology_two, original_dataset, dirty_dataset,
                                      noised_entries, features)
    else:
        methodology_two = 0
    if combination_ratio[2] != 0:
        methodology_three = number_entries - (methodology_one + methodology_two)
        if not methodology_three < 1:
            indexes = extract_indexes(original_dataset, methodology_three, noised_entries)
            dataset_backup = dirty_dataset.copy(deep=True)
            for magnitude in range(magnitude_step, 60, magnitude_step):
                inaccurate_value_generator(original_dataset, dirty_dataset,
                                           features, magnitude, indexes)
                if not in_memory:
                    save_output(features_filename, mess_percentage, combination_ratio,
                                magnitude, dirty_dataset, flag_logs, noised_entries, 
                                methodology_one, methodology_two, logID)

                results.append(dirty_dataset)
                dirty_dataset = dataset_backup.copy(deep=True)
        else:
            if not in_memory:
                save_output(features_filename, mess_percentage, combination_ratio, 
                            0, dirty_dataset, flag_logs, noised_entries, 
                            methodology_one, methodology_two, logID)
            results.append(dirty_dataset)
    else:
        if not in_memory:
            save_output(features_filename, mess_percentage, combination_ratio, 0, 
                        dirty_dataset, flag_logs, noised_entries, 
                        methodology_one, methodology_two, logID)
        results.append(dirty_dataset)
    return results


def combination_features():
    combinations = []
    maxValueTemp = 2 ** len(features_list)
    for i in range(maxValueTemp):
        combinations.append(f'{i:b}')
        combinations[len(combinations) - 1] = combinations[len(combinations) - 1].zfill(len(features_list))
    return combinations


def noise_combinations():
    noise_ratio = []
    noise_combinations = []
    for y in range(0, 100 + ratio_step, ratio_step):
        noise_ratio.append(y)
    for first_elem in noise_ratio:
        for second_element in noise_ratio:
            for third_element in noise_ratio:
                if first_elem + second_element + third_element == 100:
                    """
                    Exclude the combination [0,0,0] because the clean dataset case is already addressed in 
                    features combinations dimension
                    """
                    if first_elem != 0 or second_element != 0 or third_element != 0:
                        noise_combinations.append([first_elem, second_element, third_element])
    return noise_combinations


def select_features(combination):
    extracted_features = []
    for x in range(len(combination)):
        if combination[x] == '1':
            extracted_features.append(features_list[x])
    return extracted_features


def noising_single_run(params):
    logging.basicConfig(filename=logging_folder + "noising.log", level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s: %(message)s')
    features, mess_percentage, combination_ratio, original_dataset,logID = params

    noisingID = filename_generation(features, mess_percentage, combination_ratio, "#")
    logging.info("[START NOISING], experiment %s - %s", logID, noisingID)

    duration_noised_datasets = []
    if "duration_phase" in features:
        duration_noised_datasets = noising_orchestrator(original_dataset, mess_percentage,
                                                        combination_ratio, ["duration_phase"],
                                                        features, in_memory, True, logID)
    logs_by_case = []
    if not duration_noised_datasets == []:
        for duration_noised_dataset in duration_noised_datasets:
            log_by_case = format_dataset_by_incidents(duration_noised_dataset,"incident_id")
            logs_by_case.append(log_by_case)
    else:
        log_by_case = format_dataset_by_incidents(original_dataset,"incident_id")
        logs_by_case.append(log_by_case)

    if len(duration_noised_datasets) == 0 and not len(logs_by_case) == 1:
        raise Exception("Sorry, no numbers below zero")
    elif len(duration_noised_datasets) > 1 and not len(duration_noised_datasets) == len(
            logs_by_case):
        raise Exception("Sorry, no numbers below zero")

    for log_by_case in logs_by_case:
        param_feat = features.copy()
        param_feat_filename = param_feat.copy()
        if "duration_phase" in param_feat:
            param_feat.remove("duration_phase")
        noised_datasets = noising_orchestrator(log_by_case, mess_percentage,
                                               combination_ratio, param_feat, param_feat_filename,
                                               in_memory, False, logID)
    
    logging.info("[END NOISING], experiment %s - %s", logID, noisingID)


def noising_main(original_dataset):
    logpath = original_dataset.split("/")
    logID = logpath[len(logpath)-1].replace(".csv","")
    
    combinations_f = combination_features()
    combination_ratios = noise_combinations()

    noising_params = []
    for combination_feature in combinations_f:
        if int(combination_feature) != 0:
            features = select_features(combination_feature)
            """
            The combination without error is yet been created when we do not mess any feature, so is possible start
            noising from the first not zero noising percentage
            """
            for mess_percentage in range(messing_step, 51, messing_step):
                for combination_ratio in combination_ratios:
                    noising_params.append([features, mess_percentage, combination_ratio, original_dataset,logID])
    
    with ProcessPool(max_workers=config.num_cores) as pool:
        process = pool.map(noising_single_run, noising_params)