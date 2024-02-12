import os, warnings, logging
import pandas as pd
from pebble import ProcessPool

warnings.filterwarnings('ignore')

import config
from bmk_modules.augmentation import augment_log
from bmk_modules.sampling import sample_log
from bmk_modules.utils import format_dataset_by_incidents
from bmk_modules.run_models import cost_computation
from bmk_modules.evaluation import compare_models, multi_metric_ranks
from bmk_modules.noising import noising_main, validate_logs_files, validate_logs_by_case
from bmk_modules.analysis import main_analysis

loggingfolder=config.logging_folder

input_logfolder=config.input_logfolder
tmp_folder=config.tmp_folder
output_folder=config.output_folder
output_cleanfolder=config.output_cleanfolder
output_cleanmetrics=config.output_cleanmetrics
output_cleanranks=config.output_cleanranks
plot_folder=config.plot_folder

output_noisedfolder=config.output_noisedfolder
output_noisedlog_folder=config.output_noisedlog_folder
output_noisedlogbycase_folder=config.output_noisedlogbycase_folder
output_noisedresult_folder=config.output_noisedresult_folder
output_noisemetrics=config.output_noisemetrics
output_noiseranks=config.output_noiseranks

perform_augmentation=config.perform_augmentation
perform_sampling=config.perform_sampling
sampling_percentage=config.sampling_percentage

features_augment=config.features_augment
only_validation=config.only_validation
valOut_folder=config.valOut_folder

def run_single_benchmark(params):
    inputlog_event, output_noisefolder = params
    output_metrics = output_noisefolder+output_noisemetrics
    output_ranks = output_noisefolder+output_noiseranks    

    logging.info("[START experiment]: %s - %s",inputlog_event, output_noisefolder)
    df_enrichednoiselog = cost_computation(inputlog_event,output_noisefolder)
    df_metrics = compare_models(df_enrichednoiselog,"incident_id", output_metrics)
    multi_metric_ranks(df_metrics,output_ranks.replace(".csv",""))
    logging.info("[END experiment]: %s - %s",inputlog_event, output_noisefolder)

if __name__ == '__main__':
    if not os.path.exists(loggingfolder): os.makedirs(loggingfolder)
    logging.basicConfig(filename=loggingfolder+"main_bmk.log", level=logging.DEBUG, 
                        filemode="w", format='%(asctime)s - %(levelname)s: %(message)s')
    
    if not os.path.exists(tmp_folder): os.mkdir(tmp_folder)
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    if not os.path.exists(output_cleanfolder): os.mkdir(output_cleanfolder)
    if not os.path.exists(output_noisedfolder): os.mkdir(output_noisedfolder)
    if not os.path.exists(output_noisedresult_folder): os.mkdir(output_noisedresult_folder)

    # actual_input_folder=input_logfolder
    # if perform_sampling:
    #     logging.info("[START SAMPLING]")
    #     for logfile in os.listdir(input_logfolder):
    #         sample_log(input_logfolder+logfile,sampling_percentage,tmp_folder)
    #         logging.info("Sampled: %s", logfile)
        
    #     actual_input_folder=tmp_folder
    #     logging.info("[END SAMPLING]")
    # else:
    #     actual_input_folder=input_logfolder
    #     logging.info("No sampling")

    # if perform_augmentation:
    #     logging.info("[START AUGMENTATION]")
    #     considered_feat = []
    #     count_sample=1
    #     for feature in features_augment:
    #         feature_target = feature
    #         features_training = [ele for ele in features_augment if ele != feature]
    #         for logfile in os.listdir(actual_input_folder):
    #             if "0" not in logfile: continue
    #             augment_log(actual_input_folder+logfile, actual_input_folder+"IMPlog"+str(count_sample)+".csv", 
    #                         features_training, feature_target)
    #             logging.info("Augmented %s, target: %s", logfile, feature)
    #             count_sample+=1
    #     logging.info("[END AUGMENTATION]")
    # else:
    #     logging.info("No augmentation")
    
    # actual_input_folder=tmp_folder
    # for logsampled in os.listdir(actual_input_folder):
    #     logging.info("[START CLEAN CASE] %s", logsampled)
    #     log_by_case = format_dataset_by_incidents(actual_input_folder+logsampled, "incident_id")
    #     df_enrichedcleanlog = cost_computation(actual_input_folder+logsampled,output_cleanfolder)
    #     df_metrics = compare_models(df_enrichedcleanlog,"incident_id",output_cleanmetrics.replace(".csv",logsampled))
    #     test = multi_metric_ranks(df_metrics,output_cleanranks.replace(".csv",logsampled))
    #     logging.info("[END CLEAN CASE] %s", logsampled)

    # for logfile in os.listdir(actual_input_folder):
    #     logging.info("[START NOISING] %s", logfile)
    #     noising_main(actual_input_folder+logfile)
    #     logging.info("[END NOISING] %s", logfile)

    # params_bmk=[]
    # for noisedlogfolder in os.listdir(output_noisedlog_folder):
    #     for noisedlog in os.listdir(output_noisedlog_folder+noisedlogfolder):
    #         if "csv" not in noisedlog: continue
    #         input_log = output_noisedlog_folder+noisedlogfolder+"/"+noisedlog
    #         noise_id = noisedlog.replace(".csv","")

    #         result_folderlog = output_noisedresult_folder+noisedlogfolder+"/"
    #         result_folder = result_folderlog+noise_id+"/"
    #         if not os.path.exists(result_folderlog): os.mkdir(result_folderlog)
    #         if not os.path.exists(result_folder): os.mkdir(result_folder)

    #         params_bmk.append([input_log, result_folder])
    
    # with ProcessPool(max_workers=config.num_cores) as pool:
    #     process = pool.map(run_single_benchmark, params_bmk)


    # ### Validate Noising
    # for logfile in os.listdir(actual_input_folder):
    #     original_dataset = actual_input_folder+logfile
    #     validate_logs_files(original_dataset, 1, only_validation, output_noisedlog_folder, valOut_folder)
    #     original_logByCase = format_dataset_by_incidents(original_dataset,"incident_id")
    #     validate_logs_by_case(original_logByCase, 1, only_validation, output_noisedlogbycase_folder, valOut_folder)


    ### Analysis
    if not os.path.exists(plot_folder): os.mkdir(plot_folder)
    main_analysis()