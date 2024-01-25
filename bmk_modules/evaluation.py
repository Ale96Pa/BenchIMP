import os, statistics
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import config

cutoff = config.cutoff
cutoff_factor = config.cutoff_factor

def normalize_df_col(df, cols):
    for c in cols:
        df[c] = (df[c]-df[c].min()) / (df[c].max()-df[c].min())
    return df

def compare_models(enriched_log, case_k, outfilename):
    results=[]
    considered_gts=[]
    cols = [col for col in enriched_log.columns if 'gt' in col]

    for gt_k in cols:
        df_normal_model = normalize_df_col(enriched_log, [gt_k])

        for gt_h in cols:
            if gt_k != gt_h and gt_h not in considered_gts:
                param_dict={}
                df_normal_gt = normalize_df_col(enriched_log, [gt_h])
                
                y = df_normal_model[gt_k].to_list()
                x = df_normal_gt[gt_h].to_list()
                # y = enriched_log[gt_k].to_list()
                # x = enriched_log[gt_h].to_list()
                mse = metrics.mean_squared_error(y, x)
                mae = metrics.mean_absolute_error(y, x)
                mad = metrics.median_absolute_error(y, x)
                param_dict["model-gt"] = gt_k+"-"+gt_h
                param_dict["model"] = gt_k
                param_dict["gt"] = gt_h
                param_dict["mse"] = mse
                param_dict["mae"] = mae
                param_dict["mad"] = mad
                results.append(param_dict)
                considered_gts.append(gt_k)
    
    resultsDf = pd.DataFrame(results)
    resultsDf.to_csv(outfilename, index=False)
    return resultsDf

    # df_log = log_by_case

    # results = []
    # with open(parameter_file) as csv_r:
    #     reader = csv.DictReader(csv_r)
    #     for param_dict in reader:
    #         df_model = compute_cost(log_by_case, case_k, param_dict)
            
    #         # df_log['incident_id'] = df_log['incident_id'].astype(str)
    #         # df_model['incident_id'] = df_model['incident_id'].astype(str)
    #         # df = pd.merge(df_log[[case_k, cost_k]], df_model, how='inner', on=[case_k])

    #         # df = normalize_df_col(df, [cost_k, "cost_model"])

    #         if len(df)>0:
    #             y = df[cost_k].to_list()
    #             x = df["cost_model"].to_list()
    #             mse = metrics.mean_squared_error(y, x)
    #             mae = metrics.mean_absolute_error(y, x)
    #             mad = metrics.median_absolute_error(y, x)
    #             param_dict["mse"] = mse
    #             param_dict["mae"] = mae
    #             param_dict["mad"] = mad
    #             results.append(param_dict)

    # resultsDf = pd.DataFrame(results)
    # # resultsDf.to_csv("./results.csv")
    # with open(parameter_file, 'w', newline='') as param_obj:            
    #     dict_writer = csv.DictWriter(param_obj, results[0].keys())
    #     dict_writer.writeheader()
    #     dict_writer.writerows(results)
    # return resultsDf

MseClassification = {
    "model-gt": [],
    "Model": [],
    "Gt": [],
    "mse_n": []
}
MaeClassification = {
    "model-gt": [],
    "Model": [],
    "Gt": [],
    "mae_n": []
}
MedianErrorClassification = {
    "model-gt": [],
    "Model": [],
    "Gt": [],
    "median_err_n": []
}
ScoresClassification = {
    "model-gt": [],
    "Model": [],
    "Gt": [],
    "MeasuresArray": [],
    "WeightsArray": [],
    "ScoresArray": [],
    "FinalScore": [],
    "FinalPosition": [],
    "MaximumRelativeNormalizedScore": [],
    "MaximumAbsoluteNormalizedScore": [],
    "CutFactorFlag": []
}
PositionsClassification = {
    "model-gt": [],
    "Model": [],
    "Gt": [],
    "MeasuresArray": [],
    "ScoresArray": [],
    "FinalScore": [],
    "FinalPosition": [],
    "MaximumRelativeNormalizedScore": [],
    "MaximumAbsoluteNormalizedScore": []
}
AggregationRankingPositionsByModel = {
    "Model": [],
    "PositionsArray": [],
    "Median": [],
    "Mean": [],
    "VariationField": [],
    "Deviance": [],
    "Variance": [],
    "StandardDeviation": [],
    "CoefficientVariation": []
}
AggregationRankingScoresByModel = {
    "Model": [],
    "ScoresArray": [],
    "Median": [],
    "Mean": [],
    "VariationField": [],
    "Deviance": [],
    "Variance": [],
    "StandardDeviation": [],
    "CoefficientVariation": []
}

distance_scores = []
# colors_models = ["#66c2a5", "#8da0cb", "#fc8d62"]
# model_names = ["LR", "ETR", "CP"]

def tss(a):
    m = statistics.mean(a)
    n = 0
    for i in a:
        n += ((i-m)**2)
    return (n)


def ExtractAggregationByModelRankingPositions():
    for x in range(PositionsClassification["Model"].__len__()):
        if AggregationRankingPositionsByModel["Model"].__contains__(PositionsClassification["Model"][x]):
            indexModello = AggregationRankingPositionsByModel["Model"].index(PositionsClassification["Model"][x])
            AggregationRankingPositionsByModel["PositionsArray"][indexModello].append(PositionsClassification["FinalScore"][x])
        else:
            AggregationRankingPositionsByModel["Model"].append(PositionsClassification["Model"][x])
            AggregationRankingPositionsByModel["PositionsArray"].append([PositionsClassification["FinalScore"][x]])

    for x in range(AggregationRankingPositionsByModel["Model"].__len__()):
        AggregationRankingPositionsByModel["Median"].append(statistics.median(AggregationRankingPositionsByModel["PositionsArray"][x]))
        media = statistics.mean(AggregationRankingPositionsByModel["PositionsArray"][x])
        AggregationRankingPositionsByModel["Mean"].append(media)
        AggregationRankingPositionsByModel["VariationField"].append(max(AggregationRankingPositionsByModel["PositionsArray"][x])-min(AggregationRankingPositionsByModel["PositionsArray"][x]))
        if len(AggregationRankingPositionsByModel["PositionsArray"][x])<2: AggregationRankingPositionsByModel["PositionsArray"][x]+=AggregationRankingPositionsByModel["PositionsArray"][x]
        # ArrayValues = np.asarray(AggregationRankingPositionsByModel["PositionsArray"][x]).astype(np.float64)
        AggregationRankingPositionsByModel["StandardDeviation"].append(statistics.stdev(AggregationRankingPositionsByModel["PositionsArray"][x], media))
        
        AggregationRankingPositionsByModel["Deviance"].append(tss(AggregationRankingPositionsByModel["PositionsArray"][x]))
        AggregationRankingPositionsByModel["Variance"].append(statistics.variance(AggregationRankingPositionsByModel["PositionsArray"][x], media))
        coefficienteVariazioneFormula = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
        coefficienteVariazione = coefficienteVariazioneFormula(statistics.variance(AggregationRankingPositionsByModel["PositionsArray"][x]))
        AggregationRankingPositionsByModel["CoefficientVariation"].append(coefficienteVariazione)
def ExtractAggregationByModelRankingScores():
    for x in range(ScoresClassification["Model"].__len__()):
        if AggregationRankingScoresByModel["Model"].__contains__(ScoresClassification["Model"][x]):
            indexModello = AggregationRankingScoresByModel["Model"].index(ScoresClassification["Model"][x])
            AggregationRankingScoresByModel["ScoresArray"][indexModello].append(ScoresClassification["FinalScore"][x])
        else:
            AggregationRankingScoresByModel["Model"].append(ScoresClassification["Model"][x])
            AggregationRankingScoresByModel["ScoresArray"].append([ScoresClassification["FinalScore"][x]])

    for x in range(AggregationRankingScoresByModel["Model"].__len__()):
        AggregationRankingScoresByModel["Median"].append(statistics.median(AggregationRankingScoresByModel["ScoresArray"][x]))
        media = statistics.mean(AggregationRankingScoresByModel["ScoresArray"][x])
        AggregationRankingScoresByModel["Mean"].append(media)
        AggregationRankingScoresByModel["VariationField"].append(max(AggregationRankingScoresByModel["ScoresArray"][x])-min(AggregationRankingScoresByModel["ScoresArray"][x]))
        if len(AggregationRankingScoresByModel["ScoresArray"][x])<2: AggregationRankingScoresByModel["ScoresArray"][x]+=AggregationRankingScoresByModel["ScoresArray"][x]
        # ArrayValues = np.asarray(AggregationRankingScoresByModel["ScoresArray"][x]).astype(np.float64)
        AggregationRankingScoresByModel["StandardDeviation"].append(statistics.stdev(AggregationRankingScoresByModel["ScoresArray"][x], media))
        AggregationRankingScoresByModel["Deviance"].append(tss(AggregationRankingScoresByModel["ScoresArray"][x]))
        AggregationRankingScoresByModel["Variance"].append(statistics.variance(AggregationRankingScoresByModel["ScoresArray"][x], media))
        coefficienteVariazioneFormula = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
        coefficienteVariazione = coefficienteVariazioneFormula(statistics.variance(AggregationRankingScoresByModel["ScoresArray"][x]))
        AggregationRankingScoresByModel["CoefficientVariation"].append(coefficienteVariazione)

def getIndexScenario(scenario, df):
    count = 0

    for index, row in df.iterrows():
        if row['model-gt'] == scenario:
            return count
        count += 1
    return -1

def createPositionBasedRanking(Punteggi, misure):
    PosizioneClassifica = 1
    IndexPareggi = []
    MassimoRelativo = 0
    MassimoAssoluto = MseClassification["model-gt"].__len__() + \
                      MaeClassification["model-gt"].__len__() + \
                      MedianErrorClassification["model-gt"].__len__()

    for x in range(0, Punteggi[0]["model-gt"].__len__()):
        if not PositionsClassification["model-gt"].__contains__(Punteggi[0]["model-gt"][x]):
            PositionsClassification["model-gt"].append(Punteggi[0]["model-gt"][x])
            PositionsClassification["Model"].append(Punteggi[0]['Model'][x])
            PositionsClassification["Gt"].append(Punteggi[0]['Gt'][x])
            PositionsClassification["MeasuresArray"].append(misure)
            PositionsClassification["ScoresArray"].append(0)
            PositionsClassification["FinalScore"].append(0)
            PositionsClassification["FinalPosition"].append(0)
            PositionsClassification["MaximumAbsoluteNormalizedScore"].append(0)
            PositionsClassification["MaximumRelativeNormalizedScore"].append(0)

    for x in range(Punteggi.__len__()):
        for y in range(Punteggi[x]["model-gt"].__len__()):
            Parimerito = True
            if(y != Punteggi[x]["model-gt"].__len__()-1):
                if Punteggi[x][misure[x]][y] != Punteggi[x][misure[x]][y + 1]:
                    Parimerito = False
            else:
                Parimerito = False
            if not Parimerito:
                IndexElemento = PositionsClassification["model-gt"].index(Punteggi[x]["model-gt"][y])
                IndexPareggi.append(IndexElemento)

                SaveEvenScores(IndexPareggi, PosizioneClassifica, PositionsClassification)

                IndexPareggi = []
                PosizioneClassifica = PosizioneClassifica + 1
            else:
                IndexPareggi.append(
                    PositionsClassification["model-gt"].index(Punteggi[x]["model-gt"][y]))
        MassimoRelativo += PosizioneClassifica
        PosizioneClassifica = 1

    calculateFinalScore()

    for x in range(PositionsClassification["model-gt"].__len__()):
        PositionsClassification["MaximumAbsoluteNormalizedScore"][x] = \
            1 - (PositionsClassification["FinalScore"][x] / MassimoAssoluto)
        PositionsClassification["MaximumRelativeNormalizedScore"][x] = \
            1 - (PositionsClassification["FinalScore"][x] / MassimoRelativo)

    sortRankPosition()
    calculateFinalPositionRanking()

def calculateFinalScore():
    for x in range(PositionsClassification["model-gt"].__len__()):
        for y in range(PositionsClassification["ScoresArray"][x].__len__()):
            PositionsClassification["FinalScore"][x] += PositionsClassification["ScoresArray"][x][y]

def calculateFinalPositionRanking():
    PosizioneClassifica = 1
    IndexPareggi = []
    for y in range(PositionsClassification["model-gt"].__len__()):
        Parimerito = True
        if (y != PositionsClassification["model-gt"].__len__() - 1):
            if PositionsClassification["FinalScore"][y] != PositionsClassification["FinalScore"][y + 1]:
                Parimerito = False
        else:
            Parimerito = False
        if not Parimerito:
            IndexElemento = y
            IndexPareggi.append(IndexElemento)

            SaveEvenScoreFinalScore(IndexPareggi, PosizioneClassifica, PositionsClassification)

            IndexPareggi = []
            PosizioneClassifica = PosizioneClassifica + 1
        else:
            IndexPareggi.append(y)
    PosizioneClassifica = 1

def SaveEvenScores(Indexes, SommatoriaPunteggi, Classifica):
    for x in range(Indexes.__len__()):
        if Classifica["ScoresArray"][Indexes[x]] == 0:
            Classifica["ScoresArray"][Indexes[x]] = [SommatoriaPunteggi]
        else:
            Classifica["ScoresArray"][Indexes[x]].append(SommatoriaPunteggi)

def SaveEvenScoreFinalScore(Indexes, SommatoriaPunteggi, Classifica):
    for x in range(Indexes.__len__()):
            Classifica["FinalPosition"][Indexes[x]] = SommatoriaPunteggi

def sortRankPosition():
    for x in range(PositionsClassification["model-gt"].__len__() - 1):
        for y in range(x + 1, PositionsClassification["model-gt"].__len__()):
                if PositionsClassification["FinalScore"][x] > PositionsClassification["FinalScore"][y]:
                    temp1 = PositionsClassification['model-gt'][x]
                    temp2 = PositionsClassification['Model'][x]
                    temp3 = PositionsClassification['Gt'][x]
                    temp4 = PositionsClassification["MeasuresArray"][x]
                    temp5 = PositionsClassification["ScoresArray"][x]
                    temp6 = PositionsClassification["FinalScore"][x]
                    temp7 = PositionsClassification["MaximumAbsoluteNormalizedScore"][x]
                    temp8 = PositionsClassification["MaximumRelativeNormalizedScore"][x]

                    PositionsClassification['model-gt'][x] = PositionsClassification['model-gt'][y]
                    PositionsClassification['Model'][x] = PositionsClassification['Model'][y]
                    PositionsClassification['Gt'][x] = PositionsClassification['Gt'][y]
                    PositionsClassification["MeasuresArray"][x] = PositionsClassification["MeasuresArray"][y]
                    PositionsClassification["ScoresArray"][x] = PositionsClassification["ScoresArray"][y]
                    PositionsClassification["FinalScore"][x] = PositionsClassification["FinalScore"][y]
                    PositionsClassification["MaximumAbsoluteNormalizedScore"][x] = \
                    PositionsClassification["MaximumAbsoluteNormalizedScore"][y]
                    PositionsClassification["MaximumRelativeNormalizedScore"][x] = \
                    PositionsClassification["MaximumRelativeNormalizedScore"][y]

                    PositionsClassification['model-gt'][y] = temp1
                    PositionsClassification['Model'][y] = temp2
                    PositionsClassification['Gt'][y] = temp3
                    PositionsClassification["MeasuresArray"][y] = temp4
                    PositionsClassification["ScoresArray"][y] = temp5
                    PositionsClassification["FinalScore"][y] = temp6
                    PositionsClassification["MaximumAbsoluteNormalizedScore"][y] = temp7
                    PositionsClassification["MaximumRelativeNormalizedScore"][y] = temp8

def createEditBasedWeights(dfs, pesi):
    indexPeso = 0
    MassimoAssoluto = sum(pesi)
    for df in dfs:
        for index, row in df.iterrows():
            scenario = row['model-gt']
            misura = df.columns[1]
            if not ScoresClassification["model-gt"].__contains__(scenario):
                ScoresClassification["model-gt"].append(scenario)
                ScoresClassification["Model"].append(row["Model"])
                ScoresClassification["Gt"].append(row["Gt"])
                ScoresClassification["ScoresArray"].append([row[3] * pesi[indexPeso]])
                ScoresClassification["MeasuresArray"].append([misura])
                ScoresClassification["WeightsArray"].append([pesi[indexPeso]])
                ScoresClassification["MaximumRelativeNormalizedScore"].append(0)
                ScoresClassification["MaximumAbsoluteNormalizedScore"].append(0)
                ScoresClassification["FinalPosition"].append(0)
                ScoresClassification["CutFactorFlag"].append("Sotto")
            else:
                index = ScoresClassification["model-gt"].index(scenario)
                ScoresClassification["ScoresArray"][index].append(row[3] * pesi[indexPeso])
                ScoresClassification["MeasuresArray"][index].append(misura)
                ScoresClassification["WeightsArray"][index].append(pesi[indexPeso])
        indexPeso += 1

    MassimoRelativo = 0
    for x in range(ScoresClassification["model-gt"].__len__()):
        Sommatoria = 0
        for y in range(ScoresClassification["ScoresArray"][x].__len__()):
            Sommatoria += ScoresClassification["ScoresArray"][x][y]
        ScoresClassification["FinalScore"].append(Sommatoria)
        if Sommatoria > MassimoRelativo:
            MassimoRelativo = Sommatoria

    for x in range(ScoresClassification["model-gt"].__len__()):
        ScoresClassification["FinalScore"][x] = round(ScoresClassification["FinalScore"][x], cutoff_factor)
        if not MassimoAssoluto == 0:
            ScoresClassification["MaximumAbsoluteNormalizedScore"][x] = round(1 - (ScoresClassification["FinalScore"][x]/MassimoAssoluto), cutoff_factor)
        else:
            ScoresClassification["MaximumAbsoluteNormalizedScore"][x] = 0
        if not MassimoRelativo == 0:
            ScoresClassification["MaximumRelativeNormalizedScore"][x] = round(1 - (ScoresClassification["FinalScore"][x] / MassimoRelativo), cutoff_factor)
        else:
            ScoresClassification["MaximumRelativeNormalizedScore"][x] = 0

    calculateFinalScoreRanking()
    sortWeightRanking()

def calculateFinalScoreRanking():
    PosizioneClassifica = 1
    IndexPareggi = []
    for y in range(ScoresClassification["model-gt"].__len__()):
        Parimerito = True
        if (y != ScoresClassification["model-gt"].__len__() - 1):
            if ScoresClassification["FinalScore"][y] != ScoresClassification["FinalScore"][y + 1]:
                Parimerito = False
        else:
            Parimerito = False
        if not Parimerito:
            IndexElemento = y
            IndexPareggi.append(IndexElemento)

            SaveEvenScoreFinalScore(IndexPareggi, PosizioneClassifica, ScoresClassification)

            IndexPareggi = []
            PosizioneClassifica = PosizioneClassifica + 1
        else:
            IndexPareggi.append(y)
    PosizioneClassifica = 1

def sortWeightRanking():
    for x in range(ScoresClassification["model-gt"].__len__() - 1):
        for y in range(x + 1, ScoresClassification["model-gt"].__len__()):
            if ScoresClassification["FinalScore"][x] > ScoresClassification["FinalScore"][y]:
                temp1 = ScoresClassification['model-gt'][x]
                temp2 = ScoresClassification['Model'][x]
                temp3 = ScoresClassification['Gt'][x]
                temp4 = ScoresClassification["MeasuresArray"][x]
                temp5 = ScoresClassification["ScoresArray"][x]
                temp6 = ScoresClassification["FinalScore"][x]
                temp7 = ScoresClassification["MaximumAbsoluteNormalizedScore"][x]
                temp8 = ScoresClassification["MaximumRelativeNormalizedScore"][x]

                ScoresClassification['model-gt'][x] = ScoresClassification['model-gt'][y]
                ScoresClassification['Model'][x] = ScoresClassification['Model'][y]
                ScoresClassification['Gt'][x] = ScoresClassification['Gt'][y]
                ScoresClassification["MeasuresArray"][x] = ScoresClassification["MeasuresArray"][y]
                ScoresClassification["ScoresArray"][x] = ScoresClassification["ScoresArray"][y]
                ScoresClassification["FinalScore"][x] = ScoresClassification["FinalScore"][y]
                ScoresClassification["MaximumAbsoluteNormalizedScore"][x] = ScoresClassification["MaximumAbsoluteNormalizedScore"][y]
                ScoresClassification["MaximumRelativeNormalizedScore"][x] = ScoresClassification["MaximumRelativeNormalizedScore"][y]

                ScoresClassification['model-gt'][y] = temp1
                ScoresClassification['Model'][y] = temp2
                ScoresClassification['Gt'][y] = temp3
                ScoresClassification["MeasuresArray"][y] = temp4
                ScoresClassification["ScoresArray"][y] = temp5
                ScoresClassification["FinalScore"][y] = temp6
                ScoresClassification["MaximumAbsoluteNormalizedScore"][y] = temp7
                ScoresClassification["MaximumRelativeNormalizedScore"][y] = temp8

def rename_models(model_ids):
    names=[]
    for mod in model_ids:
        if "<" in mod:
            names.append(mod.replace("<1","<linearReg").replace("<2","<extraTree").replace("<3","<causalCost").replace("<4","<causalAll"))
        else:
            names.append(mod.replace("1","linearReg").replace("2","extraTree").replace("3","causalCost").replace("4","causalAll"))
    return names

def sortModel(dati, measure, ascending):
    for x in range(dati["model-gt"].__len__() - 1):
        for y in range(x + 1, dati["model-gt"].__len__()):
            if ascending == False:
                if dati[measure][x] < dati[measure][y]:
                    temp = dati['model-gt'][x]
                    temp2 = dati['Model'][x]
                    temp3 = dati['Gt'][x]
                    temp4 = dati[measure][x]

                    dati['model-gt'][x] = dati['model-gt'][y]
                    dati['Model'][x] = dati['Model'][y]
                    dati['Gt'][x] = dati['Gt'][y]
                    dati[measure][x] = dati[measure][y]

                    dati['model-gt'][y] = temp
                    dati['Model'][y] = temp2
                    dati['Gt'][y] = temp3
                    dati[measure][y] = temp4
            else:
                if dati[measure][x] > dati[measure][y]:
                    temp = dati['model-gt'][x]
                    temp2 = dati['Model'][x]
                    temp3 = dati['Gt'][x]
                    temp4 = dati[measure][x]

                    dati['model-gt'][x] = dati['model-gt'][y]
                    dati['Model'][x] = dati['Model'][y]
                    dati['Gt'][x] = dati['Gt'][y]
                    dati[measure][x] = dati[measure][y]

                    dati['model-gt'][y] = temp
                    dati['Model'][y] = temp2
                    dati['Gt'][y] = temp3
                    dati[measure][y] = temp4


def sortAggregateModels():
    for x in range(AggregationRankingPositionsByModel["Model"].__len__() - 1):
        for y in range(x + 1, AggregationRankingPositionsByModel["Model"].__len__()):
            if AggregationRankingPositionsByModel["Median"][x] > AggregationRankingPositionsByModel["Median"][y]:
                temp1 = AggregationRankingPositionsByModel["Model"][x]
                temp2 = AggregationRankingPositionsByModel["PositionsArray"][x]
                temp3 = AggregationRankingPositionsByModel["Median"][x]
                AggregationRankingPositionsByModel["Model"][x] = AggregationRankingPositionsByModel["Model"][y]
                AggregationRankingPositionsByModel["PositionsArray"][x] = AggregationRankingPositionsByModel["PositionsArray"][y]
                AggregationRankingPositionsByModel["Median"][x] = AggregationRankingPositionsByModel["Median"][y]
                AggregationRankingPositionsByModel["Model"][y] = temp1
                AggregationRankingPositionsByModel["PositionsArray"][y] = temp2
                AggregationRankingPositionsByModel["Median"][y] = temp3

    for x in range(AggregationRankingScoresByModel["Model"].__len__() - 1):
        for y in range(x + 1, AggregationRankingScoresByModel["Model"].__len__()):
            if AggregationRankingScoresByModel["Median"][x] > AggregationRankingScoresByModel["Median"][y]:
                temp1 = AggregationRankingScoresByModel["Model"][x]
                temp2 = AggregationRankingScoresByModel["ScoresArray"][x]
                temp3 = AggregationRankingScoresByModel["Median"][x]
                AggregationRankingScoresByModel["Model"][x] = AggregationRankingScoresByModel["Model"][y]
                AggregationRankingScoresByModel["ScoresArray"][x] = AggregationRankingScoresByModel["ScoresArray"][y]
                AggregationRankingScoresByModel["Median"][x] = AggregationRankingScoresByModel["Median"][y]
                AggregationRankingScoresByModel["Model"][y] = temp1
                AggregationRankingScoresByModel["ScoresArray"][y] = temp2
                AggregationRankingScoresByModel["Median"][y] = temp3


def CalculateCutCut():
    for x in range(ScoresClassification["model-gt"].__len__()-1):
        for y in range(x+1, ScoresClassification["model-gt"].__len__(), 1):
            DistanzaPunteggi = abs(ScoresClassification["FinalScore"][x] - ScoresClassification["FinalScore"][y])
            distance_scores.append(DistanzaPunteggi)
    global cutoff
    cutoff = np.mean(distance_scores)


def CalculatePositionRespectCutFactor():
    for x in range(ScoresClassification["model-gt"].__len__()):
        distanza = ScoresClassification["FinalScore"][x] - ScoresClassification["FinalScore"][0]
        if distanza < cutoff:
            ScoresClassification["CutFactorFlag"][x] = "Sotto"
        else:
            ScoresClassification["CutFactorFlag"][x] = "Sopra"

def reset_rankings():
    MseClassification['model-gt'] = []
    MseClassification['Model'] = []
    MseClassification['Gt'] = []
    MaeClassification['model-gt'] = []
    MaeClassification['Model'] = []
    MaeClassification['Gt'] = []
    MedianErrorClassification['model-gt'] = []
    MedianErrorClassification['Model'] = []
    MedianErrorClassification['Gt'] = []
    MseClassification['mse_n'] = []
    MaeClassification['mae_n'] = []
    MedianErrorClassification['median_err_n'] = []
    ScoresClassification["model-gt"] = []
    ScoresClassification["Model"] = []
    ScoresClassification["Gt"] = []
    ScoresClassification["MeasuresArray"] = []
    ScoresClassification["WeightsArray"] = []
    ScoresClassification["ScoresArray"] = []
    ScoresClassification["FinalScore"] = []
    ScoresClassification["FinalPosition"] = []
    ScoresClassification["MaximumRelativeNormalizedScore"] = []
    ScoresClassification["MaximumAbsoluteNormalizedScore"] = []
    ScoresClassification["CutFactorFlag"] = []
    PositionsClassification["model-gt"] = []
    PositionsClassification["Model"] = []
    PositionsClassification["Gt"] = []
    PositionsClassification["MeasuresArray"] = []
    PositionsClassification["ScoresArray"] = []
    PositionsClassification["FinalScore"] = []
    PositionsClassification["FinalPosition"] = []
    PositionsClassification["MaximumRelativeNormalizedScore"] = []
    PositionsClassification["MaximumAbsoluteNormalizedScore"] = []
    AggregationRankingPositionsByModel["Model"] = []
    AggregationRankingPositionsByModel["PositionsArray"] = []
    AggregationRankingPositionsByModel["Median"] = []
    AggregationRankingPositionsByModel["Mean"] = []
    AggregationRankingPositionsByModel["VariationField"] = []
    AggregationRankingPositionsByModel["Deviance"] = []
    AggregationRankingPositionsByModel["Variance"] = []
    AggregationRankingPositionsByModel["StandardDeviation"] = []
    AggregationRankingPositionsByModel["CoefficientVariation"] = []
    AggregationRankingScoresByModel["Model"] = []
    AggregationRankingScoresByModel["ScoresArray"] = []
    AggregationRankingScoresByModel["Median"] = []
    AggregationRankingScoresByModel["Mean"] = []
    AggregationRankingScoresByModel["VariationField"] = []
    AggregationRankingScoresByModel["Deviance"] = []
    AggregationRankingScoresByModel["Variance"] = []
    AggregationRankingScoresByModel["StandardDeviation"] = []
    AggregationRankingScoresByModel["CoefficientVariation"] = []

def multi_metric_ranks(dfMetrics, output_path):
    if not os.path.exists(output_path): os.makedirs(output_path)
    reset_rankings()
    
    for index, row in dfMetrics.iterrows():
        scenarioEstratto = row['model-gt']
        componentiScenario = scenarioEstratto.split("-")
        scenario = componentiScenario[0]+"-"+componentiScenario[1]
        MseClassification['model-gt'].append(scenario)
        MseClassification['Model'].append(componentiScenario[0])
        MseClassification['Gt'].append(componentiScenario[1])

        MaeClassification['model-gt'].append(scenario)
        MaeClassification['Model'].append(componentiScenario[0])
        MaeClassification['Gt'].append(componentiScenario[1])

        MedianErrorClassification['model-gt'].append(scenario)
        MedianErrorClassification['Model'].append(componentiScenario[0])
        MedianErrorClassification['Gt'].append(componentiScenario[1])

        MseClassification['mse_n'].append(round(row['mse'], cutoff_factor))
        MaeClassification['mae_n'].append(round(row['mae'], cutoff_factor))
        MedianErrorClassification['median_err_n'].append(round(row['mad'], cutoff_factor))

    sortModel(MseClassification, "mse_n", False)
    sortModel(MaeClassification, "mae_n", False)
    sortModel(MedianErrorClassification, "median_err_n", False)


    ClassificaMseDf = pd.DataFrame(MseClassification)
    ClassificaMaeDf = pd.DataFrame(MaeClassification)
    ClassificaMedianErrorDf = pd.DataFrame(MedianErrorClassification)

    ClassificaMseDf.to_csv(f"{output_path}/mseRank.csv", index=False)
    ClassificaMaeDf.to_csv(f"{output_path}/maeRank.csv", index=False)
    ClassificaMedianErrorDf.to_csv(f"{output_path}/medianRank.csv", index=False)
    #endregion

    createPositionBasedRanking([ClassificaMseDf, ClassificaMaeDf, ClassificaMedianErrorDf], ["mse_n", "mae_n", "median_err_n"])

    ClassificaPosizioneDf = pd.DataFrame(PositionsClassification)
    ClassificaPosizioneDf.to_csv(f"{output_path}/positionRank.csv", index=False)

    alfa = 0.33
    beta = 0.33
    gamma = 0.33
    createEditBasedWeights([ClassificaMseDf, ClassificaMaeDf, ClassificaMedianErrorDf], [alfa, beta, gamma])

    ExtractAggregationByModelRankingPositions()
    ExtractAggregationByModelRankingScores()

    sortAggregateModels()

    PosizioniAggregataModelloDf = pd.DataFrame(AggregationRankingPositionsByModel)
    PunteggiAggregataModelloDf = pd.DataFrame(AggregationRankingScoresByModel)

    CalculateCutCut()
    CalculatePositionRespectCutFactor()

    classificaPesiDf = pd.DataFrame(ScoresClassification)

    classificaPesiDf.to_csv(f"{output_path}/weightRank.csv", index=False)
    PosizioniAggregataModelloDf.to_csv(f"{output_path}/positionRankModels.csv", index=False)
    PunteggiAggregataModelloDf.to_csv(f"{output_path}/scoreRankModels.csv", index=False)
