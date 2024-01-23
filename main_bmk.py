import glob, os, sys, warnings, logging
import pandas as pd
from pebble import ProcessPool

warnings.filterwarnings('ignore')

import config
from bmk_modules.sampling import sample_log
from bmk_modules.utils import format_dataset_by_incidents,filename_generation
from bmk_modules.run_models import cost_computation
from bmk_modules.evaluation import compare_models

loggingfolder=config.logging_folder

input_logfolder=config.input_logfolder
tmp_folder=config.tmp_folder
output_folder=config.output_folder
output_cleanfolder=config.output_cleanfolder
output_cleanmetrics=config.output_cleanmetrics

perform_augmentation=config.perform_augmentation
perform_sampling=config.perform_sampling
sampling_percentage=config.sampling_percentage

if __name__ == '__main__':
    if not os.path.exists(loggingfolder): os.makedirs(loggingfolder)
    logging.basicConfig(filename=loggingfolder+"main_bmk", level=logging.DEBUG, filemode="w",
                        format='%(asctime)s - %(levelname)s: %(message)s')
    
    if not os.path.exists(tmp_folder): os.mkdir(tmp_folder)
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(output_cleanfolder): os.mkdir(output_cleanfolder)

    if perform_augmentation:
        logging.info("[START AUGMENTATION]")
        
        logging.info("[END AUGMENTATION]")
    else:
        logging.info("No augmentation")

    if perform_sampling:
        logging.info("[START SAMPLING]")
        for logfile in os.listdir(input_logfolder):
            sampled_dataset = sample_log(input_logfolder+logfile,sampling_percentage,tmp_folder)
            logging.info("Sampled: %s", logfile)
        logging.info("[END SAMPLING]")
    else:
        logging.info("No sampling")

    for logsampled in os.listdir(tmp_folder):
        if "0" in logsampled: continue
        log_by_case = format_dataset_by_incidents(tmp_folder+logsampled, "incident_id")
        test_name = filename_generation([], 0, [0,0,0], 0)
        df_enrichedcleanlog = cost_computation(tmp_folder+logsampled,log_by_case,output_cleanfolder)
        df_metrics = compare_models(df_enrichedcleanlog,"incident_id",output_cleanmetrics)
        print(df_metrics)



    # false_noise_gt_dfFull = df_full
    # logging.info("[CLEAN CASE] Ground Truths calculated")

    # result = models.models_main(df_full, log_by_case, test_name, root_path)
    # logging.info("[CLEAN CASE] models run")

    # classification.classification_main(result, test_name, root_path)
    # logging.info("[CLEAN CASE] models assessed")
        



    # # Create Noised LogByCase
    # noising.noising_orchestrator(original_dataset, alignment_file)

    # files_csv = glob.glob(f"{output_logByCase}/*.csv")
    # # Declared here to avoid useless rework

    # params_bmk = []
    # for flag_noise_gt in [False, True]:
    #     if flag_noise_gt:
    #         root_path = conf.output_gts_noised
    #     else:
    #         root_path = conf.output_gts_clear
    #     for file_csv in files_csv:
    #         file_csv = file_csv.replace("\\", "/")
    #         subfolders = file_csv.replace(".csv", "").split("/")
    #         filename_csv = subfolders[len(subfolders) - 1]

    #         if not flag_noise_gt:
    #             df_full = false_noise_gt_dfFull
    #         else:
    #             df_full = pd.DataFrame
    #         params_bmk.append([flag_noise_gt, file_csv, filename_csv, df_full, root_path, original_dataset])
    # logging.info("---Parameters collected: START BENCHMARKING ---")
    # with ProcessPool(max_workers=conf.num_cores) as pool:
    #     process = pool.map(run_single_benchmark, params_bmk)

    # Extract Final Results
    # try:
    #     result_extractor.results_module_main()
    #     alignment_file = pd.read_csv(f"{alignment_file_path}")
    #     original_dataset = pd.read_csv(util_dataset)
    #     # Validate Noising
    #     noise_validator.validate_logs_files(original_dataset, 1, only_validation, files_path_logs, valOutPath)
    #     original_logByCase = format_dataset.format_dataset_by_incidents(original_dataset, alignment_file,
    #                                                                     "incident_id")
    #     noise_validator.validate_logs_by_case(original_logByCase, 1, only_validation, files_path_logByCase, valOutPath)
    # except Exception as e:
    #     logging.error("[ERROR in result analysis] %s", e)