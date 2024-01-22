import glob, os, sys, warnings, logging
from pebble import ProcessPool

# warnings.filterwarnings('ignore')

import config

logfolder = config.logging_folder

perform_augmentation=config.perform_augmentation
perform_sampling=config.perform_sampling

if __name__ == '__main__':
    if not os.path.exists(logfolder): os.makedirs(logfolder)
    logging.basicConfig(filename=logfolder+"main_bmk", level=logging.DEBUG, filemode="w",
                        format='%(asctime)s - %(levelname)s: %(message)s')
    
    if perform_augmentation:
        logging.info("[START AUGMENTATION]")
    else:
        logging("No augmentation")

    if perform_sampling:
        logging.info("[START SAMPLING]")
    else:
        logging.info("No sampling")

    # if perform_sampling:
    #     logging.info("[START SAMPLING]")
    #     try:
    #         input_dataset = pd.read_csv(input_logs, sep=";")
    #         incidents_original = len(input_dataset['incident_id'].unique())
    #         input_dataset = sampling.sampling_orchestrator(input_dataset, sampling_percentage)
    #         input_dataset.to_csv(sampled_log, index=False)
    #         incidents_sampled = len(input_dataset['incident_id'].unique())
    #         logging.info(f"Original number of incidents {incidents_original}")
    #         logging.info(f"Sampling percentage {sampling_percentage}")
    #         logging.info(f"Samples number of incidents {incidents_sampled}")
    #     except Exception as e:
    #         logging.error("[ERROR in input sampling] %s", e)
    #         sys.exit("Exit with error")
    # else:
    #     try:
    #         input_dataset = pd.read_csv(input_logs, sep=";")
    #         input_dataset.to_csv(sampled_log, index=False)
    #     except Exception as e:
    #         logging.error("[ERROR in input sampling %s]", e)
    #         sys.exit("Exit with error")
    #     logging.info("[SAMPLING] already computed")

    # if perform_preproc:
    #     logging.info("[START PREPROCESSING]")
    #     try:
    #         input_dataset = pd.read_csv(sampled_log)
    #         preproc.perform_preproc(input_dataset)
    #     except Exception as e:
    #         logging.error("[ERROR in input data processing] %s", e)
    #         sys.exit("Exit with error")
    #     try:
    #         original_dataset = pd.read_csv(util_dataset)
    #     except Exception as e:
    #         logging.error("[ERROR in original data processing] %s", e)
    #         sys.exit("Exit with error")

    #     logging.info("[END PREPROCESSING]")
    # else:
    #     try:
    #         original_dataset = pd.read_csv(util_dataset, sep=";")
    #     except Exception as e:
    #         logging.error("[ERROR in original data processing] %s", e)
    #         sys.exit('Something bad happened')
    #     logging.info("[PREPROCESSING] already computed")

    # alignment_file = pd.read_csv(f"{alignment_file_path}")
    # false_noise_gt_dfFull = pd.DataFrame()
    # try:
    #     # Clean case
    #     log_by_case = format_dataset.format_dataset_by_incidents(original_dataset, alignment_file,
    #                                                              "incident_id")
    #     test_name = filename_generation.filename_generation([], 0, [0, 0, 0], 0)
    #     root_path = conf.output_gts_clear
    #     logging.info("[CLEAN CASE] dataset formatted and created")

    #     df_full = gt.ground_thruths_main(original_dataset, log_by_case, "", root_path)
    #     false_noise_gt_dfFull = df_full
    #     logging.info("[CLEAN CASE] Ground Truths calculated")

    #     result = models.models_main(df_full, log_by_case, test_name, root_path)
    #     logging.info("[CLEAN CASE] models run")

    #     classification.classification_main(result, test_name, root_path)
    #     logging.info("[CLEAN CASE] models assessed")
    # except Exception as e:
    #     logging.error("[ERROR in alignment file processing] %s", e)
    #     sys.exit("Exit with error")

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