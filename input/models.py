import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

class Models:

    """
    Ground Truth 1 (GT1)
    Cost of each trace according to [Kieninger]
    [Kieninger] Axel Kieninger, Florian Berghoff, Hansj ̈org Fromm, and Gerhard Satzger.
    Simulation-Based Quantification of Business Impacts Caused by Service Incidents.
    """
    def gt1(log, log_by_case, case_k, fract=1):
        extra_work={}
        case_ids = set(log[case_k].to_list())
        grouped_by_case = log.groupby([case_k])
        for caseID in case_ids:
            single_case_df = grouped_by_case.get_group(caseID)
            trace_id = single_case_df[case_k].to_list()[0]
            trace = log.query(case_k + " == '" + str(trace_id) + "'")["event"].to_list()

            numN = abs(trace.count('N')-1)
            numA = abs(trace.count('A')-2)
            numW = abs(trace.count('W')-1)
            numR = abs(trace.count('R')-1)
            numC = abs(trace.count('C')-1)

            extra_work[caseID] = [len(trace),sum([numN,numA,numW,numR,numC])]
        
        l_dict = []
        for index, row in log_by_case.iterrows():
            num_employee = row['reassignment_count'] + 1
            duration = row['duration_process']
            incID = row[case_k]

            tot_activities = extra_work[incID][0]
            extra_activities = extra_work[incID][1]

            cost = ((int(num_employee) * duration)/tot_activities)*extra_activities
            l_dict.append({case_k: row[case_k], "gt1": cost * fract})
        return pd.DataFrame(l_dict)

    """
    Ground Truth 2 (GT2)
    Cost of traces for SLA violations according to [Moura]
    [Moura] MOURA, Antão, et al. A quantitative approach to IT investment allocation to 
    improve business results. In: Seventh IEEE International Workshop on Policies 
    for Distributed Systems and Networks (POLICY'06). IEEE, 2006. p. 9 pp.-95.
    """
    def gt2(log, log_by_case, case_k, fract=1):
        extra_work={}
        case_ids = set(log[case_k].to_list())
        grouped_by_case = log.groupby([case_k])
        for caseID in case_ids:
            single_case_df = grouped_by_case.get_group(caseID)
            trace_id = single_case_df[case_k].to_list()[0]
            trace = log.query(case_k + " == '" + str(trace_id) + "'")["event"].to_list()

            numN = abs(trace.count('N')-1)
            numA = abs(trace.count('A')-2)
            numW = abs(trace.count('W')-1)
            numR = abs(trace.count('R')-1)
            numC = abs(trace.count('C')-1)

            extra_work[caseID] = [len(trace),sum([numN,numA,numW,numR,numC])]
        
        l_dict = []
        for index, row in log_by_case.iterrows():
            duration = row['duration_process']
            priority = row['preproc_priority']
            incID = row[case_k]

            tot_activities = extra_work[incID][0]
            extra_activities = extra_work[incID][1]

            cost = (duration/tot_activities)*extra_activities*priority
            l_dict.append({case_k: row[case_k], "gt2": cost * fract})
        return pd.DataFrame(l_dict)


    """
    Ground Truth 3 (GT3)
    Cost of the trace based on person-hours metric according to [Dumas]
    [Dumas] Marlon Dumas, Marcello La Rosa, Jan Mendling, Hajo A Reijers, et al.
    Fundamentals of business process management, volume 1. Springer, 2013.
    """
    def gt3(log_by_case, case_k, fract=1):
        l_dict = []
        for index, row in log_by_case.iterrows():
            duration_process = row['duration_process']
            num_employee = row['reassignment_count'] + 1
            cost = round(duration_process * num_employee)
            l_dict.append({case_k: row[case_k], "gt3": cost * fract})
        return pd.DataFrame(l_dict)


    """
    Ground Truth 4 (GT4)
    Cost of the traces according to the model defined in [4]
    [4] Sasha Romanosky. Examining the costs and causes of cyber incidents.
    Journal of Cybersecurity, page tyw001, August 2016.
    """
    def gt4(log_by_case, case_k, fract=1):
        l_dict = []
        for index, row in log_by_case.iterrows():
            num_employee = row['reassignment_count'] + 1
            impact = row['preproc_impact']
            cost = 2.67 * int(num_employee) * impact
            l_dict.append({case_k: row[case_k], "gt4": cost * fract})
        return pd.DataFrame(l_dict)
    
    """
    Emulation of a newly introduced assessment model
    """
    def etr(log_by_case, case_k):
        df = log_by_case
        y = df["duration_process"]
        X = df[["reassignment_count","preproc_urgency"]]
        model = ExtraTreesRegressor(n_estimators=100).fit(X, y)
        features = model.feature_importances_

        l_dict = []
        for index, row in log_by_case.iterrows():
            num_employee = row['reassignment_count'] + 1
            urgency = row['preproc_urgency']
            cost = (int(num_employee) * features[0])+(int(urgency) * features[1])
            l_dict.append({case_k: row[case_k], "gtETR": cost})
        return pd.DataFrame(l_dict)