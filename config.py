"""
Benchmark parameters
"""
num_cores = 10

sampling_percentage = 20

perform_sampling = True
perform_augmentation = True

noising_flag = False
in_memory = False

"""
File Management parameters
"""
input_folder = 'input/'
input_logfolder=input_folder+"logs/"
input_reallog = input_logfolder+"IMPlog.csv"
input_models = input_folder+"models.py"

tmp_folder = input_folder+"smp/"

output_folder = 'output/'
output_cleanfolder = output_folder+"clean/"
output_cleanmetrics = output_cleanfolder+"metrics.csv"
output_cleanranks = output_cleanfolder+"ranks.csv"

output_noisedfolder = output_folder+"noise/"
output_noisedlogbycase_folder = output_noisedfolder+"logbycase/"
output_noisedlog_folder = output_noisedfolder+"log/"
output_noisedresult_folder = output_noisedfolder+"results/"

output_noisemetrics = "metrics.csv"
output_noiseranks = "ranks.csv"

# parameters_gts = f'models_parameters'
# output_path_models = f"models"
# ground_truth_file = f"gts"
# bmk_output = f"benchmark"
# output_path_classification = f"classification"
# path_clean_classification = f"{output_gts_clear}classification/D0_P0_R0_I0_U0_P0_[V0,N0,M0(0)]"

"""
Logging
"""
logging_folder = "logging/"

"""
Multi-Metric Analysis
"""
cutoff = 20
cutoff_factor = 3


"""
Noising
"""
noise_gt = False
ratio_step = 20
messing_step = 25
magnitude_step = 25
features_list = ["duration_phase", "preproc_priority", "reassignment_count", "preproc_impact", "preproc_urgency"]
features_type = ["float", "enum", "integer", "enum", "enum"]
features_augment = ["duration_process", "preproc_priority", "reassignment_count", "preproc_impact", "preproc_urgency"]

valOut_folder =  output_folder+"noisingValidation/"
noising_validation_output = valOut_folder+"noisingValidOutput/"
noising_validation_report = valOut_folder+"noisingValidReport/"
only_validation = True


"""
Analysis
"""
colors_models = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']




# """
# Preprocessing and performance parameters
# """
# features = ['category', 'made_sla', 'knowledge', 'u_priority_confirmation', 'priority',
#             'reassignment_count', 'reopen_count']
# date_format = '%d/%m/%Y %H:%M'
# noMagnitudeTypes = ["enum", "boolean"]
# # alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]



# # """
# # Classification
# # """
# # round_factor = 20
# # round = 0

# """
# Noise Validation Setup
# """
# noising_validation_output = "./NoisingValidationOutput"
# noising_validation_report = "./NoisingValidationReport"
# only_validation = True
# files_path_logs = "./Output/NoisedLogByCase"
# files_path_logByCase = "./Output/NoisedLogs"
# valOut_path = "./OutputNoisingValidation"

# """
# Analyses Setup
# """
# path_clean_gt = "OutputCleanGt/benchmark/GtFull.csv"
# # path_noised_gts = f"{output_gts_noised}benchmark"
# flag_noised_gt_values = [False]
# metrics = ["mae", "mse", "median"]
# models_name = ["LR", "ETR", "CP"]
# gts_name = ["Gt1", "Gt2", "Gt3", "Gt4", "Gt5"]
# colors_model = ["#66c2a5", "#8da0cb", "#fc8d62"]
# colors_gt = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
# features = ["duration_phase", "preproc_priority", "reassignment_count", "preproc_impact", "preproc_urgency"]