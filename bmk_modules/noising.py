import logging, random, os, glob, re
import numpy as np
import pandas as pd
from pebble import ProcessPool
from matplotlib import pyplot as plt

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
    if flag_logs:
        output_path = output_noisedLogs+logID+"/"
    else:
        output_path = output_logByCase+logID+"/"
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


def noising_orchestrator(original_dataset, mess_percentage, combination_ratio, 
                         features, in_memory: bool, flag_logs, logID):
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
                    save_output(features, mess_percentage, combination_ratio,
                                magnitude, dirty_dataset, flag_logs, noised_entries, 
                                methodology_one, methodology_two, logID)

                results.append(dirty_dataset)
                dirty_dataset = dataset_backup.copy(deep=True)
        else:
            if not in_memory:
                save_output(features, mess_percentage, combination_ratio, 
                            0, dirty_dataset, flag_logs, noised_entries, 
                            methodology_one, methodology_two, logID)
            results.append(dirty_dataset)
    else:
        if not in_memory:
            save_output(features, mess_percentage, combination_ratio, 0, 
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
    features, mess_percentage, combination_ratio, original_dataset,logID = params

    noisingID = filename_generation(features, mess_percentage, combination_ratio, "#")
    logging.info("[START NOISING], experiment %s - %s", logID, noisingID)

    original_df = pd.read_csv(original_dataset)
    duration_noised_datasets = noising_orchestrator(original_df, mess_percentage,
                        combination_ratio, features, in_memory, True, logID)
    
    # duration_noised_datasets = []
    # if "duration_phase" in features:
    #     original_df = pd.read_csv(original_dataset)
    #     duration_noised_datasets = noising_orchestrator(original_df, mess_percentage,
    #                                                     combination_ratio, ["duration_phase"],
    #                                                     features, in_memory, True, logID)

    # logs_by_case = []
    # if not duration_noised_datasets == []:
    #     for duration_noised_dataset in duration_noised_datasets:
    #         log_by_case = format_dataset_by_incidents(duration_noised_dataset,"incident_id")
    #         logs_by_case.append(log_by_case)
    # else:
    #     log_by_case = format_dataset_by_incidents(original_dataset,"incident_id")
    #     logs_by_case.append(log_by_case)

    # if len(duration_noised_datasets) == 0 and not len(logs_by_case) == 1:
    #     logging.error("Sorry, no numbers below zero")
    #     raise Exception("Sorry, no numbers below zero")
    # elif len(duration_noised_datasets) > 1 and not len(duration_noised_datasets) == len(
    #         logs_by_case):
    #     logging.error("Sorry, no numbers below zero")
    #     raise Exception("Sorry, no numbers below zero")

    # for log_by_case in logs_by_case:
    #     param_feat = features.copy()
    #     param_feat_filename = param_feat.copy()
    #     if "duration_phase" in param_feat: 
    #         param_feat = [ele for ele in param_feat if ele != "duration_phase"]
    #     noised_datasets = noising_orchestrator(log_by_case, mess_percentage,
    #                                            combination_ratio, param_feat,
    #                                            in_memory, False, logID)
    logging.info("[END NOISING], experiment %s - %s", logID, noisingID)


def noising_main(original_dataset):
    logging.basicConfig(filename=logging_folder + "noising.log", level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s: %(message)s')
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


########################## Noising Validation
validation_output = config.noising_validation_output
validation_report = config.noising_validation_report


class output_model:
    def __init__(self):
        self.test = []
        self.noising_percentage = []
        self.noising_class = []
        self.magnitude_class = []
        self.null_value_percentage = []
        self.null_class = []
        self.incorrect_assignment_percentage = []
        self.incorrect_class = []
        self.inaccurate_assignment_percentage = []
        self.inaccurate_class = []

    def add_result(self, test, noising_percentage, magnitude_class, null_percentage,
                   incorrect_percentage, inaccurate_percentage, wrong_test_report):
        self.test.append(test)
        self.noising_percentage.append(noising_percentage)
        noising_class = self.determine_class(noising_percentage)
        if not noising_class == "Correct":
            wrong_test_report.add_wrong_test(test, "Noising", noising_class)
        self.noising_class.append(noising_class)

        if not (magnitude_class == "Correct" or magnitude_class == "N/D"):
            wrong_test_report.add_wrong_test(test, "Magnitude", noising_class)
        self.magnitude_class.append(magnitude_class)

        self.null_value_percentage.append(null_percentage)
        noising_class = self.determine_class(null_percentage)
        if not noising_class == "Correct":
            wrong_test_report.add_wrong_test(test, "NullValues", noising_class)
        self.null_class.append(noising_class)

        self.incorrect_assignment_percentage.append(incorrect_percentage)
        noising_class = self.determine_class(incorrect_percentage)
        if not noising_class == "Correct":
            wrong_test_report.add_wrong_test(test, "Incorrect", noising_class)
        self.incorrect_class.append(noising_class)

        self.inaccurate_assignment_percentage.append(inaccurate_percentage)
        noising_class = self.determine_class(inaccurate_percentage)
        if not noising_class == "Correct":
            wrong_test_report.add_wrong_test(test, "Inaccurate", noising_class)
        self.inaccurate_class.append(noising_class)

    def out_dataframe(self):
        df = pd.DataFrame()
        df['test'] = self.test
        df['noising_percentage'] = self.noising_percentage
        df['noising_class'] = self.noising_class
        df['magnitude_class'] = self.magnitude_class
        df['null_value_percentage'] = self.null_value_percentage
        df['null_class'] = self.null_class
        df['incorrect_assignment_percentage'] = self.incorrect_assignment_percentage
        df['incorrect_class'] = self.incorrect_class
        df['inaccurate_assignment_percentage'] = self.inaccurate_assignment_percentage
        df['inaccurate_class'] = self.inaccurate_class
        return df

    @staticmethod
    def determine_class(value):
        if value == 1:
            return "Correct"
        elif 0.8 <= value < 1:
            return "Quite Correct"
        elif 0.5 <= value < 0.8:
            return "Wrong"
        else:
            return "Seriously Wrong"


class report_model:
    def __init__(self):
        self.NoisingType = []
        self.Max = []
        self.Min = []
        self.Median = []
        self.Mean = []
        self.FirstQuartile = []
        self.ThirdQuartile = []
        self.PercentageCorrect = []
        self.PercentageQuiteCorrect = []
        self.PercentageWrong = []
        self.PercentageSeriouslyWrong = []
        self.MagnitudePercentageCorrect = []
        self.MagnitudePercentageQuiteCorrect = []
        self.MagnitudePercentageWrong = []
        self.MagnitudePercentageSeriouslyWrong = []

    def add_result(self, type, max, min, median, mean, first_quartile, third_quartile, correct_percentage,
                   quite_percentage, wrong_percentage, seriously_wrong_percentage, magnitude_correct_percentage,
                   magnitude_quite_percentage, magnitude_wrong_percentage, magnitude_seriously_wrong_percentage):
        self.NoisingType.append(type)
        self.Max.append(max)
        self.Min.append(min)
        self.Median.append(median)
        self.Mean.append(mean)
        self.FirstQuartile.append(first_quartile)
        self.ThirdQuartile.append(third_quartile)
        self.PercentageCorrect.append(correct_percentage)
        self.PercentageQuiteCorrect.append(quite_percentage)
        self.PercentageWrong.append(wrong_percentage)
        self.PercentageSeriouslyWrong.append(seriously_wrong_percentage)
        self.MagnitudePercentageCorrect.append(magnitude_correct_percentage)
        self.MagnitudePercentageQuiteCorrect.append(magnitude_quite_percentage)
        self.MagnitudePercentageWrong.append(magnitude_wrong_percentage)
        self.MagnitudePercentageSeriouslyWrong.append(magnitude_seriously_wrong_percentage)

    def out_dataframe(self):
        df = pd.DataFrame()
        df['Noising Methodology'] = self.NoisingType
        df['Mean'] = self.Mean
        df['Min'] = self.Min
        df['Max'] = self.Max
        df['Median'] = self.Median
        df['FirstQuartile'] = self.FirstQuartile
        df['ThirdQuartile'] = self.ThirdQuartile
        df['PercentageCorrect'] = self.PercentageCorrect
        df['PercentageQuiteCorrect'] = self.PercentageQuiteCorrect
        df['PercentageWrong'] = self.PercentageWrong
        df['PercentageSeriouslyWrong'] = self.PercentageSeriouslyWrong
        df['MagnitudePercentageCorrect'] = self.MagnitudePercentageCorrect
        df['MagnitudePercentageQuiteCorrect'] = self.MagnitudePercentageQuiteCorrect
        df['MagnitudePercentageWrong'] = self.MagnitudePercentageWrong
        df['MagnitudePercentageSeriouslyWrong'] = self.MagnitudePercentageSeriouslyWrong
        return df


class wrong_tests_report_model:
    def __init__(self):
        self.NoisingType = ["Noising Methodology", "Magnitude", "Null Assignment", "Incorrect Assignment",
                            "Inaccurate Assignment"]
        self.QuiteCorrectTests = [[], [], [], [], []]
        self.WrongTests = [[], [], [], [], []]
        self.SeriouslyWrongTests = [[], [], [], [], []]

    def add_wrong_test(self, test, methodology, error_class):
        match methodology:
            case 'Noising':
                match error_class:
                    case 'Quite Correct':
                        self.QuiteCorrectTests[0].append(test)
                    case 'Wrong':
                        self.WrongTests[0].append(test)
                    case "Seriously Wrong":
                        self.SeriouslyWrongTests[0].append(test)
            case 'Magnitude':
                match error_class:
                    case 'Quite Correct':
                        self.QuiteCorrectTests[1].append(test)
                    case 'Wrong':
                        self.WrongTests[1].append(test)
                    case "Seriously Wrong":
                        self.SeriouslyWrongTests[1].append(test)
            case 'NullValues':
                match error_class:
                    case 'Quite Correct':
                        self.QuiteCorrectTests[2].append(test)
                    case 'Wrong':
                        self.WrongTests[2].append(test)
                    case "Seriously Wrong":
                        self.SeriouslyWrongTests[2].append(test)
            case 'Incorrect':
                match error_class:
                    case 'Quite Correct':
                        self.QuiteCorrectTests[3].append(test)
                    case 'Wrong':
                        self.WrongTests[3].append(test)
                    case "Seriously Wrong":
                        self.SeriouslyWrongTests[3].append(test)
            case _:
                match error_class:
                    case 'Quite Correct':
                        self.QuiteCorrectTests[4].append(test)
                    case 'Wrong':
                        self.WrongTests[4].append(test)
                    case "Seriously Wrong":
                        self.SeriouslyWrongTests[4].append(test)

    def out_dataframe(self):
        df = pd.DataFrame()
        df['Noising Methodology'] = self.NoisingType
        df['Quite Correct'] = self.QuiteCorrectTests
        df['WrongTests'] = self.WrongTests
        df['SeriouslyWrongTests'] = self.SeriouslyWrongTests
        return df


def extract_domain_values(feature, dataset):
    values = []
    dataset_values = dataset[feature].values.tolist()
    for value in dataset_values:
        if value not in values:
            values.append(value)
    return values


def plot_histogram(tests, results, graph_name, experiment, only_verification, valOut_path):
    casi = tests
    punteggi = results

    fig = plt.figure(figsize=(30, 15))

    # creating the bar plot
    plt.scatter(casi, punteggi, color='blue')
    if (only_verification is False):
        if not os.path.exists(f"./exp{experiment+1}/plots"):
            os.makedirs(f"./exp{experiment+1}/plots")
        plt.savefig(f"./exp{experiment+1}/plots/{graph_name}.png")
    else:
        if not os.path.exists(f"{valOut_path}/plots"):
            os.makedirs(f"{valOut_path}/plots")
        plt.savefig(f"{valOut_path}/plots/{graph_name}.png")
    plt.close(fig)


def plot_boxgraph(output: output_model, log_flag, experiment, only_verification, valOut_path):
    labels = ["Data Noising", "Null Values", "Incorrect Values", "Imprecise Values"]
    values = [output.noising_percentage, output.null_value_percentage,
              output.incorrect_assignment_percentage, output.inaccurate_assignment_percentage]

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.xlabel("Noising Methodology")
    plt.ylabel("Percentages")
    plt.title("Boxplot Noising Percentages")
    plt.boxplot(values, notch=True, sym=None, labels=labels)

    # creating the bar plot
    if(only_verification is False):
        if not os.path.exists(f"./exp{experiment+1}/plots"):
            os.makedirs(f"./exp{experiment+1}/plots")
        if log_flag is True:
            graph_name = "ReportBoxPlotLogs"
        else:
            graph_name = "ReportBoxPlotLogByCase"
        plt.savefig(f"./exp{experiment+1}/plots/{graph_name}.png")
    else:
        if not os.path.exists(f"{valOut_path}/plots"):
            os.makedirs(f"{valOut_path}/plots")
        if log_flag is True:
            graph_name = "ReportBoxPlotLogs"
        else:
            graph_name = "ReportBoxPlotLogByCase"
        plt.savefig(f"{valOut_path}/plots/{graph_name}.png")
    plt.close(fig)


def validate_logs_files(original_dataset, experiment, only_verification, files_path, valOut_path):
    output_validation = output_model()
    wrong_test_report = wrong_tests_report_model()
    if(only_verification is False):
        files_csv = glob.glob(f"./exp{experiment+1}/Output/DurationNoisedDatasets/*.csv")
    else:
        files_csv = glob.glob(f"{files_path}/*.csv")
    for index in range(len(files_csv)):
        file_csv = pd.read_csv(files_csv[index])

        filename_csv = files_csv[index].replace("\\", "/")
        subfolders = filename_csv.split("/")
        filename_csv = subfolders[len(subfolders) - 1]

        test = filename_csv[0: len(filename_csv) - 4]
        filename_txt = f"{filename_csv[0:len(filename_csv) - 4]}.txt"
        if (only_verification is False):
            file_txt = open(f"./exp{experiment+1}/Output/DurationNoisedDatasets/{filename_txt}", "r")
        else:
            file_txt = open(f"{files_path}/{filename_txt}", "r")
        noised_indexes = file_txt.readlines()
        start_parameters = filename_csv.index('[')
        start_magnitude = filename_csv.index('(')
        end_magnitude = filename_csv.index(')')
        comma_indexes = []
        for match in re.finditer(r',', filename_csv):
            comma_indexes.append(match.start())
        new_line_indexes = []
        for noised_index in noised_indexes:
            for match in re.finditer(r'\n', noised_index):
                new_line_indexes.append(match.start())
        indexes_strategy_one = []
        indexes_strategy_two = []
        indexes_strategy_three = []
        for i in range(len(noised_indexes)):
            noised_index = noised_indexes[i]
            noising_strategy = noised_index[0]
            match noising_strategy:
                case 'V':
                    indexes_strategy_one.append(int(noised_index[3:new_line_indexes[i]]))
                case 'N':
                    indexes_strategy_two.append(int(noised_index[3:new_line_indexes[i]]))
                case _:
                    indexes_strategy_three.append(int(noised_index[3:new_line_indexes[i]]))
        combination_one = filename_csv[start_parameters + 2:comma_indexes[0]]
        combination_two = filename_csv[comma_indexes[0] + 2:comma_indexes[1]]
        percentage = filename_csv[16:start_parameters - 1]
        number_of_records = len(file_csv['incident_id'])
        elements_to_noise = round((int(percentage) * number_of_records) / 100)
        sum_indexes = len(indexes_strategy_one) + len(indexes_strategy_two) + len(indexes_strategy_three)
        elem_met_one = round((int(combination_one) * elements_to_noise) / 100)
        elem_met_two = round((int(combination_two) * elements_to_noise) / 100)
        elem_met_three = sum_indexes - (elem_met_one + elem_met_two)

        null_counter = 0
        for index_met_one in indexes_strategy_one:
            value = file_csv.iloc[index_met_one]['duration_phase']
            if np.isnan(value):
                null_counter += 1
        if elem_met_one == 0:
            null_percentage = 1
        else:
            null_percentage = (null_counter / elem_met_one)

        domain = extract_domain_values('duration_phase', original_dataset)
        domain_max = max(domain)
        inc_counter = 0
        for index_met_two in indexes_strategy_two:
            value = file_csv.iloc[index_met_two]['duration_phase']
            if value == 0 or value > domain_max:
                inc_counter += 1
        if elem_met_two == 0:
            inc_percentage = 1
        else:
            inc_percentage = (inc_counter / elem_met_two)

        magnitude = int(filename_csv[start_magnitude + 1:end_magnitude])
        max_magnitude_err_class = 0
        imp_counter = 0
        for index_met_three in indexes_strategy_three:
            original_value = original_dataset.iloc[index_met_three]['duration_phase']
            magnitude_step = config.magnitude_step
            min_corr_bound = original_value - ((original_value * magnitude) / 100)
            max_corr_bound = original_value + ((original_value * magnitude) / 100)
            quite_correct_min_bound = original_value - ((original_value * (magnitude + (magnitude_step / 4))) / 100)
            quite_correct_max_bound = original_value - ((original_value * (magnitude + (magnitude_step / 4))) / 100)
            wrong_min_bound = original_value - ((original_value * (magnitude + (magnitude_step / 2))) / 100)
            wrong_max_bound = original_value - ((original_value * (magnitude + (magnitude_step / 2))) / 100)
            value = file_csv.iloc[index_met_three]['duration_phase']
            if original_value == value and (not original_value == 0):
                continue
            if min_corr_bound <= value <= max_corr_bound and max_magnitude_err_class == 0:
                imp_counter += 1
                continue
            elif quite_correct_min_bound <= value <= quite_correct_max_bound and max_magnitude_err_class <= 1:
                max_magnitude_err_class = 1
                imp_counter += 1
                continue
            elif wrong_min_bound <= value <= wrong_max_bound and max_magnitude_err_class <= 2:
                max_magnitude_err_class = 2
                imp_counter += 1
                continue
            else:
                max_magnitude_err_class = 3
                imp_counter += 1
                continue

        if elem_met_three == 0:
            imp_percentage = 1
        else:
            imp_percentage = (imp_counter / elem_met_three)

        sum_noising = imp_counter + inc_counter + null_counter
        if elements_to_noise == 0:
            noising_percentage = 1
        else:
            noising_percentage = (sum_noising / elements_to_noise)

        match max_magnitude_err_class:
            case 0:
                magnitude_class = "Correct"
            case 1:
                magnitude_class = "Quite Correct"
            case 2:
                magnitude_class = "Wrong"
            case _:
                magnitude_class = "Seriously Wrong"

        output_validation.add_result(test, noising_percentage, magnitude_class, null_percentage,
                                     inc_percentage, imp_percentage, wrong_test_report)
    output_dataframe = output_validation.out_dataframe()
    if (only_verification is False):
        if not os.path.exists(f"./exp{experiment+1}/{validation_output}/"):
            os.makedirs(f"./exp{experiment+1}/{validation_output}/")
        output_dataframe.to_csv(f"./exp{experiment+1}/{validation_output}/ValidationLogs.csv", index=False)
    else:
        if not os.path.exists(f"{valOut_path}/{validation_output}/"):
            os.makedirs(f"{valOut_path}/{validation_output}/")
        output_dataframe.to_csv(f"{valOut_path}/{validation_output}/ValidationLogs.csv", index=False)
    plot_histogram(output_validation.test, output_validation.noising_percentage, 'logs_noisingPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.null_value_percentage, 'logs_nullPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.incorrect_assignment_percentage, 'logs_incorrectAssignPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.inaccurate_assignment_percentage, 'logs_inacAssignPercentage', experiment, only_verification, valOut_path)
    report = create_report(output_validation, True)
    report_dataframe = report.out_dataframe()
    wrong_test_report_dataframe = wrong_test_report.out_dataframe()
    if (only_verification is False):
        if not os.path.exists(f"./exp{experiment + 1}/{validation_report}/"):
            os.makedirs(f"./exp{experiment + 1}/{validation_report}/")
        report_dataframe.to_csv(f"./exp{experiment + 1}/{validation_report}/ReportLogs.csv", index=False)
        wrong_test_report_dataframe.to_csv(f"./exp{experiment + 1}/{validation_report}/ReportWrongTestsLogs.csv", index=False)
    else:
        if not os.path.exists(f"{valOut_path}/{validation_report}/"):
            os.makedirs(f"{valOut_path}/{validation_report}/")
        report_dataframe.to_csv(f"{valOut_path}/{validation_report}/ReportLogs.csv", index=False)
        wrong_test_report_dataframe.to_csv(f"{valOut_path}/{validation_report}/ReportWrongTestsLogs.csv", index=False)
    plot_boxgraph(output_validation, True, experiment, only_verification, valOut_path)


def create_report(output_validation, flag_logs):
    report = report_model()
    noising_percentage_mean = np.mean(output_validation.noising_percentage)
    noising_percentage_max = np.max(output_validation.noising_percentage)
    noising_percentage_min = np.min(output_validation.noising_percentage)
    noising_percentage_median = np.median(output_validation.noising_percentage)
    noising_percentage_first_quartile = np.percentile(output_validation.noising_percentage, 25)
    noising_percentage_third_quartile = np.percentile(output_validation.noising_percentage, 75)
    percentages = extract_class_percentages(output_validation.noising_class)
    report.add_result("Noising Percentile", noising_percentage_max, noising_percentage_min, noising_percentage_median,
                      noising_percentage_mean, noising_percentage_first_quartile, noising_percentage_third_quartile,
                      percentages[0], percentages[1], percentages[2], percentages[3],
                      "N/D", "N/D", "N/D", "N/D")

    null_values_mean = np.mean(output_validation.null_value_percentage)
    null_values_max = np.max(output_validation.null_value_percentage)
    null_values_min = np.min(output_validation.null_value_percentage)
    null_values_median = np.median(output_validation.null_value_percentage)
    null_values_first_quartile = np.percentile(output_validation.null_value_percentage, 25)
    null_values_third_quartile = np.percentile(output_validation.null_value_percentage, 75)
    percentages = extract_class_percentages(output_validation.null_class)
    report.add_result("Null-Value Assignment Percentile", null_values_max, null_values_min,
                      null_values_median, null_values_mean, null_values_first_quartile,
                      null_values_third_quartile, percentages[0], percentages[1], percentages[2], percentages[3],
                      "N/D", "N/D", "N/D", "N/D")

    incor_values_mean = np.mean(output_validation.incorrect_assignment_percentage)
    incor_values_max = np.max(output_validation.incorrect_assignment_percentage)
    incor_values_min = np.min(output_validation.incorrect_assignment_percentage)
    incor_values_median = np.median(output_validation.incorrect_assignment_percentage)
    incor_values_first_quartile = np.percentile(output_validation.incorrect_assignment_percentage, 25)
    incor_values_third_quartile = np.percentile(output_validation.incorrect_assignment_percentage, 75)
    percentages = extract_class_percentages(output_validation.incorrect_class)
    report.add_result("Incorrect-Value Assignment Percentile", incor_values_max, incor_values_min,
                      incor_values_median, incor_values_mean, incor_values_first_quartile,
                      incor_values_third_quartile, percentages[0], percentages[1], percentages[2], percentages[3],
                      "N/D", "N/D", "N/D", "N/D")

    if flag_logs is True:
        impr_values_mean = np.mean(output_validation.inaccurate_assignment_percentage)
        impr_values_max = np.max(output_validation.inaccurate_assignment_percentage)
        impr_values_min = np.min(output_validation.inaccurate_assignment_percentage)
        impr_values_median = np.median(output_validation.inaccurate_assignment_percentage)
        impr_values_first_quartile = np.percentile(output_validation.inaccurate_assignment_percentage, 25)
        impr_values_third_quartile = np.percentile(output_validation.inaccurate_assignment_percentage, 75)
        percentages = extract_class_percentages(output_validation.inaccurate_class)
        percentages_magnitude = extract_class_percentages(output_validation.magnitude_class)
        report.add_result("Imprecise-Value Assignment Percentile", impr_values_max, impr_values_min,
                          impr_values_median, impr_values_mean, impr_values_first_quartile,
                          impr_values_third_quartile, percentages[0], percentages[1], percentages[2], percentages[3],
                          percentages_magnitude[0], percentages_magnitude[1], percentages_magnitude[2],
                          percentages_magnitude[3])
    else:
        impr_values_mean = np.mean(output_validation.inaccurate_assignment_percentage)
        impr_values_max = np.max(output_validation.inaccurate_assignment_percentage)
        impr_values_min = np.min(output_validation.inaccurate_assignment_percentage)
        impr_values_median = np.median(output_validation.inaccurate_assignment_percentage)
        impr_values_first_quartile = np.percentile(output_validation.inaccurate_assignment_percentage, 25)
        impr_values_third_quartile = np.percentile(output_validation.inaccurate_assignment_percentage, 75)
        percentages = extract_class_percentages(output_validation.inaccurate_class)
        report.add_result("Imprecise-Value Assignment Percentile", impr_values_max, impr_values_min,
                          impr_values_median, impr_values_mean, impr_values_first_quartile,
                          impr_values_third_quartile, percentages[0], percentages[1], percentages[2], percentages[3],
                          "N/D", "N/D", "N/D", "N/D")

    return report


def extract_class_percentages(values):
    result = [0, 0, 0, 0]
    class_counts = [0, 0, 0, 0]
    for value in values:
        match value:
            case "Correct":
                class_counts[0] += 1
            case "Quite Correct":
                class_counts[1] += 1
            case "Wrong":
                class_counts[2] += 1
            case _:
                class_counts[3] += 1
    for i in range(len(result)):
        result[i] = (class_counts[i] / len(values))
    return result


def validate_logs_by_case(original_dataset, experiment, only_verification, files_path, valOut_path):
    output_validation = output_model()
    wrong_test_report = wrong_tests_report_model()
    if(only_verification is False):
        files_csv = glob.glob(f"./exp{experiment+1}/Output/NoisedLogByCase/*.csv")
    else:
        files_csv = glob.glob(f"{files_path}/*.csv")

    for index in range(len(files_csv)):
        file_csv = pd.read_csv(files_csv[index])

        filename_csv = files_csv[index].replace("\\", "/")
        subfolders = filename_csv.split("/")
        filename_csv = subfolders[len(subfolders) - 1]

        test = filename_csv[0: len(filename_csv) - 4]
        filename_txt = f"{filename_csv[0:len(filename_csv) - 4]}.txt"
        if (only_verification is False):
            file_txt = open(f"./exp{experiment+1}/Output/NoisedLogByCase/{filename_txt}", "r")
        else:
            file_txt = open(f"{files_path}/{filename_txt}", "r")
        noised_indexes = file_txt.readlines()
        start_parameters = filename_csv.index('[')
        comma_indexes = []
        for match in re.finditer(r',', filename_csv):
            comma_indexes.append(match.start())
        new_line_indexes = []
        for noised_index in noised_indexes:
            for match in re.finditer(r'\n', noised_index):
                new_line_indexes.append(match.start())
        indexes_strategy_one = []
        indexes_strategy_two = []
        indexes_strategy_three = []
        for i in range(len(noised_indexes)):
            noised_index = noised_indexes[i]
            noising_strategy = noised_index[0]
            match noising_strategy:
                case 'V':
                    indexes_strategy_one.append(int(noised_index[3:new_line_indexes[i]]))
                case 'N':
                    indexes_strategy_two.append(int(noised_index[3:new_line_indexes[i]]))
                case _:
                    indexes_strategy_three.append(int(noised_index[3:new_line_indexes[i]]))
        combination_one = filename_csv[start_parameters + 2:comma_indexes[0]]
        combination_two = filename_csv[comma_indexes[0] + 2:comma_indexes[1]]
        percentage = filename_csv[16:start_parameters - 1]
        number_of_records = len(file_csv['incident_id'])
        elements_to_noise = round((int(percentage) * number_of_records) / 100)
        sum_indexes = len(indexes_strategy_one) + len(indexes_strategy_two) + len(indexes_strategy_three)
        elem_met_one = round((int(combination_one) * elements_to_noise) / 100)
        elem_met_two = round((int(combination_two) * elements_to_noise) / 100)
        elem_met_three = sum_indexes - (elem_met_one + elem_met_two)

        # Estrazione Feature
        features = extract_features(test)

        if not len(features) == 0:
            null_counter = 0
            for index_met_one in indexes_strategy_one:
                for feature in features:
                    value = file_csv.iloc[index_met_one][feature]
                    if np.isnan(value):
                        null_counter += 1
            if elem_met_one == 0 and not null_counter > 0:
                null_percentage = 1
            else:
                total_cell_noise = len(features) * elem_met_one
                null_percentage = (null_counter / total_cell_noise)

            domains = []
            for feature in features:
                domain = extract_domain_values(feature, original_dataset)
                domains.append(domain)

            inc_counter = 0
            for index_met_two in indexes_strategy_two:
                for i in range(len(features)):
                    value = file_csv.iloc[index_met_two][features[i]]
                    if value not in domains[i]:
                        inc_counter += 1
            if elem_met_two == 0 and not inc_counter > 0:
                inc_percentage = 1
            else:
                total_cell_noise = len(features) * elem_met_two
                inc_percentage = (inc_counter / total_cell_noise)

            imp_counter = 0
            for index_met_three in indexes_strategy_three:
                for i in range(len(features)):
                    feature_type = extract_feature_type(features[i])
                    value = file_csv.iloc[index_met_three][features[i]]
                    original_value = original_dataset.iloc[index_met_three][features[i]]
                    match feature_type:
                        case 'enum':
                            if (not value == original_value) and value in domains[i]:
                                imp_counter += 1
                        case 'integer':
                            domain_min = min(domains[i])
                            domain_max = max(domains[i])
                            if (not value == original_value) and domain_min <= value <= domain_max:
                                imp_counter += 1
            if elem_met_three == 0 and not elem_met_three > 0:
                imp_percentage = 1
            else:
                total_cell_noise = len(features) * elem_met_three
                imp_percentage = (imp_counter/total_cell_noise)

            total_cell_noise = len(features) * elements_to_noise
            noised_cells = null_counter + inc_counter + imp_counter
            if total_cell_noise == 0:
                noising_percentage = 1
            else:
                noising_percentage = (noised_cells / total_cell_noise)

            output_validation.add_result(test, noising_percentage, "N/D", null_percentage,
                                         inc_percentage, imp_percentage, wrong_test_report)
    output_dataframe = output_validation.out_dataframe()
    if (only_verification is False):
        if not os.path.exists(f"./exp{experiment + 1}/{validation_output}/"):
            os.makedirs(f"./exp{experiment + 1}/{validation_output}/")
        output_dataframe.to_csv(f"./exp{experiment + 1}/{validation_output}/ValidationLogByCase.csv",
                                index=False)
    else:
        if not os.path.exists(f"{valOut_path}/{validation_output}/"):
            os.makedirs(f"{valOut_path}/{validation_output}/")
        output_dataframe.to_csv(f"{valOut_path}/{validation_output}/ValidationLogByCase.csv",
                                index=False)
    plot_histogram(output_validation.test, output_validation.noising_percentage, 'logByCase_noisingPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.null_value_percentage, 'logByCase_nullPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.incorrect_assignment_percentage, 'logByCase_incorrectAssignPercentage', experiment, only_verification, valOut_path)
    plot_histogram(output_validation.test, output_validation.inaccurate_assignment_percentage, 'logByCase_inacAssignPercentage', experiment, only_verification, valOut_path)
    report = create_report(output_validation, False)
    report_dataframe = report.out_dataframe()
    report_wrong_tests_dataframe = wrong_test_report.out_dataframe()
    if (only_verification is False):
        if not os.path.exists(f"./exp{experiment + 1}/{validation_report}/"):
            os.makedirs(f"./exp{experiment + 1}/{validation_report}/")
        report_dataframe.to_csv(f"./exp{experiment + 1}/{validation_report}/ReportLogByCase.csv", index=False)
        report_wrong_tests_dataframe.to_csv(f"./exp{experiment + 1}/{validation_report}/ReportWrongTestsLogByCase.csv", index=False)
    else:
        if not os.path.exists(f"{valOut_path}/{validation_report}/"):
            os.makedirs(f"{valOut_path}/{validation_report}/")
        report_dataframe.to_csv(f"{valOut_path}/{validation_report}/ReportLogByCase.csv", index=False)
        report_wrong_tests_dataframe.to_csv(f"{valOut_path}/{validation_report}/ReportWrongTestsLogByCase.csv", index=False)
    plot_boxgraph(output_validation, False, experiment, only_verification, valOut_path)


def extract_features(test):
    features = []
    flag_uno = test[4: 5]
    flag_due = test[7: 8]
    flag_tre = test[10: 11]
    flag_quattro = test[13: 14]
    if int(flag_uno) == 1:
        features.append(config.features_list[1])
    if int(flag_due) == 1:
        features.append(config.features_list[2])
    if int(flag_tre) == 1:
        features.append(config.features_list[3])
    if int(flag_quattro) == 1:
        features.append(config.features_list[4])
    return features


def extract_feature_type(feature):
    features_list = config.features_list
    features_type = config.features_type
    feature_index = features_list.index(feature)
    return features_type[feature_index]