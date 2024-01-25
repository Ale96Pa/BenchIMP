import random
import pandas as pd

def sample_log(input_file, sampling_percentage, outfolder):
    input_dataset = pd.read_csv(input_file)
    incidents = input_dataset['incident_id'].unique()
    extracted_incidents = []
    requested_incidents = round(len(incidents) * (sampling_percentage / 100))
    while len(extracted_incidents) < requested_incidents:
        extracted_index = random.randint(0, len(incidents) - 1)
        extracted_incident = incidents[extracted_index]
        if extracted_incident not in extracted_incidents:
            extracted_incidents.append(extracted_incident)
    deleting_indexes = []
    for index, row in input_dataset.iterrows():
        if row['incident_id'] not in extracted_incidents:
            deleting_indexes.append(index)
    clean_dataset = input_dataset.drop(input_dataset.index[deleting_indexes])
    
    outfilename = input_file.split("/")
    clean_dataset.to_csv(outfolder+outfilename[len(outfilename)-1], index=False)
    return
    # return clean_dataset