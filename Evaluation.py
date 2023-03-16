import pickle
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.stats as ss
import math
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from sklearn.metrics import classification_report,f1_score,recall_score,precision_score,accuracy_score

def getWeightedStdDev(oWeights,means,stdDevs):
    #https://math.stackexchange.com/a/4567292

    if len(oWeights) >= 10:
        n = 10
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])
        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        n6 = oWeights[indexLastN][5]
        n7 = oWeights[indexLastN][6]
        n8 = oWeights[indexLastN][7]
        n9 = oWeights[indexLastN][8]
        n10 = oWeights[indexLastN][9]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        m6 = means[indexLastN][5]
        m7 = means[indexLastN][6]
        m8 = means[indexLastN][7]
        m9 = means[indexLastN][8]
        m10 = means[indexLastN][9]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5 + n6*m6 + n7*m7 + n8*m8 + n9*m9 + n10*m10) / (n1+n2+n3+n4+n5+n6+n7+n8+n9+n10) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2 + n6*(m6-ap)**2 + n7*(m7-ap)**2 + n8*(m8-ap)**2 + n9*(m9-ap)**2 + n10*(m10-ap)**2) / (n1+n2+n3+n4+n5+n6+n7+n8+n9+n10)

    elif len(oWeights) >= 9:
        n = 9
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])
        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        n6 = oWeights[indexLastN][5]
        n7 = oWeights[indexLastN][6]
        n8 = oWeights[indexLastN][7]
        n9 = oWeights[indexLastN][8]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        m6 = means[indexLastN][5]
        m7 = means[indexLastN][6]
        m8 = means[indexLastN][7]
        m9 = means[indexLastN][8]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5 + n6*m6 + n7*m7 + n8*m8 + n9*m9) / (n1+n2+n3+n4+n5+n6+n7+n8+n9) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2 + n6*(m6-ap)**2 + n7*(m7-ap)**2 + n8*(m8-ap)**2 + n9*(m9-ap)**2) / (n1+n2+n3+n4+n5+n6+n7+n8+n9)

    elif len(oWeights) >= 8:
        n = 8
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])
        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        n6 = oWeights[indexLastN][5]
        n7 = oWeights[indexLastN][6]
        n8 = oWeights[indexLastN][7]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        m6 = means[indexLastN][5]
        m7 = means[indexLastN][6]
        m8 = means[indexLastN][7]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5 + n6*m6 + n7*m7 + n8*m8) / (n1+n2+n3+n4+n5+n6+n7+n8) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2 + n6*(m6-ap)**2 + n7*(m7-ap)**2 + n8*(m8-ap)**2) / (n1+n2+n3+n4+n5+n6+n7+n8)
        
    elif len(oWeights) >= 7:
        n = 7
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])
        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        n6 = oWeights[indexLastN][5]
        n7 = oWeights[indexLastN][6]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        m6 = means[indexLastN][5]
        m7 = means[indexLastN][6]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5 + n6*m6 + n7*m7) / (n1+n2+n3+n4+n5+n6+n7) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2 + n6*(m6-ap)**2 + n7*(m7-ap)**2) / (n1+n2+n3+n4+n5+n6+n7)
        
    elif len(oWeights) >= 6:
        n = 6
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        n6 = oWeights[indexLastN][5]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        m6 = means[indexLastN][5]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5 + n6*m6) / (n1+n2+n3+n4+n5+n6) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2 + n6*(m6-ap)**2) / (n1+n2+n3+n4+n5+n6)
        
    elif len(oWeights) >= 5:
        n = 5
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        n5 = oWeights[indexLastN][4]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        m5 = means[indexLastN][4]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4 + n5*m5) / (n1+n2+n3+n4+n5) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2 + n5*(m5-ap)**2) / (n1+n2+n3+n4+n5)

    elif len(oWeights) >= 4:
        n = 4
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])        
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        n4 = oWeights[indexLastN][3]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        m4 = means[indexLastN][3]
        ap = (n1*m1 + n2*m2 + n3*m3 + n4*m4) / (n1+n2+n3+n4) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2 + n4*(m4-ap)**2) / (n1+n2+n3+n4)

    elif len(oWeights) >= 3:
        n = 3
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])     
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        n3 = oWeights[indexLastN][2]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        m3 = means[indexLastN][2]
        ap = (n1*m1 + n2*m2 + n3*m3) / (n1+n2+n3) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2 + n3*(m3-ap)**2) / (n1+n2+n3)

    else:
        n = 2
        lastn = list(np.arange(len(oWeights))+1)[-n:]
        indexLastN = list(np.where(np.isin(ss.rankdata(oWeights,'ordinal'),lastn))[0])     
        n1 = oWeights[indexLastN][0]
        n2 = oWeights[indexLastN][1]
        m1 = means[indexLastN][0]
        m2 = means[indexLastN][1]
        ap = (n1*m1 + n2*m2) / (n1+n2) 
        stdDev = (n1*(m1-ap)**2 + n2*(m2-ap)**2) / (n1+n2)
    
    return stdDev

#%%

paramsDict = {
    'exp1':{
        'area':'E08000025',
        'poi':'School',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_School_amPeak.csv',
        'lablledSetFile':'trips_birm_School_amPeak_results.csv',
        'trainingDataFile':'birm_school_ampeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp2':{
        'area':'E08000025',
        'poi':'Hospital',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Hospital_amPeak.csv',
        'lablledSetFile':'trips_birm_Hospital_amPeak_results.csv',
        'trainingDataFile':'birm_hospital_ampeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp3':{
        'area':'E08000025',
        'poi':'Vaccination Centre',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Vaccination Centre_amPeak.csv',
        'lablledSetFile':'trips_birm_Vaccination_Centre_amPeak_results.csv',
        'trainingDataFile':'birm_vaccine_centre_ampeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp4':{
        'area':'E08000025',
        'poi':'Job Centre',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Job Centre_amPeak.csv',
        'lablledSetFile':'trips_birm_Job_Centre_amPeak_results.csv',
        'trainingDataFile':'birm_job_centre_ampeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    
    'exp5':{
        'area':'E08000026',
        'poi':'School',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_School_amPeak.csv',
        'lablledSetFile':'trips_cov_School_amPeak_results.csv',
        'trainingDataFile':'cov_school_ampeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp6':{
        'area':'E08000026',
        'poi':'Hospital',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Hospital_amPeak.csv',
        'lablledSetFile':'trips_cov_Hospital_amPeak_results.csv',
        'trainingDataFile':'cov_hospital_ampeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp7':{
        'area':'E08000026',
        'poi':'Vaccination Centre',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Vaccination Centre_amPeak.csv',
        'lablledSetFile':'trips_cov_Vaccination_Centre_amPeak_results.csv',
        'trainingDataFile':'cov_vaccine_centre_ampeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp8':{
        'area':'E08000026',
        'poi':'Job Centre',
        'stratum':'AM Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Job Centre_amPeak.csv',
        'lablledSetFile':'trips_cov_Job_Centre_amPeak_results.csv',
        'trainingDataFile':'cov_job_centre_ampeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    
    'exp9':{
        'area':'E08000025',
        'poi':'School',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_School_interPeak.csv',
        'lablledSetFile':'trips_birm_School_interPeak_results.csv',
        'trainingDataFile':'birm_school_interPeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp10':{
        'area':'E08000025',
        'poi':'Hospital',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Hospital_interPeak.csv',
        'lablledSetFile':'trips_birm_Hospital_interPeak_results.csv',
        'trainingDataFile':'birm_hospital_interPeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp11':{
        'area':'E08000025',
        'poi':'Vaccination Centre',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Vaccination Centre_interPeak.csv',
        'lablledSetFile':'trips_birm_Vaccination_Centre_interPeak_results.csv',
        'trainingDataFile':'birm_vaccine_centre_interPeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    'exp12':{
        'area':'E08000025',
        'poi':'Job Centre',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootBirm.csv',
        'OAtoOABusFile':'OAtoOABusBirm.csv',
        'unlabelledTripsFile':'trips_birm_Job Centre_interPeak.csv',
        'lablledSetFile':'trips_birm_Job_Centre_interPeak_results.csv',
        'trainingDataFile':'birm_job_centre_interPeak_training.txt',
        'OBTreeFile':'OBTreeBirm.txt',
        'urbanCentre':(52.47806970328915, -1.8989517979775046)
        },
    
    'exp13':{
        'area':'E08000026',
        'poi':'School',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_School_interPeak.csv',
        'lablledSetFile':'trips_cov_School_interPeak_results.csv',
        'trainingDataFile':'cov_school_interPeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp14':{
        'area':'E08000026',
        'poi':'Hospital',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Hospital_interPeak.csv',
        'lablledSetFile':'trips_cov_Hospital_interPeak_results.csv',
        'trainingDataFile':'cov_hospital_interPeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp15':{
        'area':'E08000026',
        'poi':'Vaccination Centre',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Vaccination Centre_interPeak.csv',
        'lablledSetFile':'trips_cov_Vaccination_Centre_interPeak_results.csv',
        'trainingDataFile':'cov_vaccine_centre_interPeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        },
    'exp16':{
        'area':'E08000026',
        'poi':'Job Centre',
        'stratum':'Inter Peak',
        'OAsReachableFile':'reachableOnFootCov.csv',
        'OAtoOABusFile':'OAtoOABusCov.csv',
        'unlabelledTripsFile':'trips_cov_Job Centre_interPeak.csv',
        'lablledSetFile':'trips_cov_Job_Centre_interPeak_results.csv',
        'trainingDataFile':'cov_job_centre_interPeak_training.txt',
        'OBTreeFile':'OBTreeCov.txt',
        'urbanCentre':(52.4078358796818, -1.5137840014354358)
        }
    }

shpFileLoc = 'E:/Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'E:/Data/oa_info.csv'

#%%

columnDict = {'GAC':{
    'mean':0,
    'std':2
    },
    'JT':{
    'mean':1,
    'std':3
        }}

#%%

demoGroups = ['lone_parent_total','age_65_to_74','cars_no_car','disability_all','emp_inactive','eth_black','eth_asian_indian']

levels = ['OD','O']
costs = ['GAC','JT']
budgets = [0.3, 0.2, 0.1, 0.07, 0.05, 0.03]
models = ['OLSRegression', 'MLPRegression', 'GNN', 'COREG', 'Mean Teacher']

results = []

resultsLoc = 'E:/Data/results/'

#exp = 'exp1'

for exp in ['exp1','exp2','exp3','exp4','exp5','exp6','exp7','exp8','exp9','exp10','exp11','exp12','exp13','exp14','exp15','exp16']:

    params = paramsDict[exp]
    
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    
    
    f = open(resultsLoc + exp +'.txt', 'rb')
    resultsDict = pickle.load(f)
    f.close()
    
    
    groundTruthOD = resultsDict[exp]['OD']['yAct']
    groundTruthO = resultsDict[exp]['O']['yAct']
    
    indexOD = resultsDict[exp]['OD']['index']
    indexO = resultsDict[exp]['O']['index']
    indexO.sort()
    
    weightsOD = resultsDict[exp]['OD']['weights']
    weightsO = resultsDict[exp]['O']['weights']
    
    # labelCostsOD = resultsDict['exp1']['OD']['labellingCosts']
    # labelCostsO = resultsDict['exp1']['O']['labellingCosts']
    
    # scalerYOD = resultsDict[exp]['OD']['scalerY']
    # scalerYO = resultsDict[exp]['O']['scalerY']
    
    oa_info = oa_info[oa_info['oa_id'].isin(indexO)]
    
    for level in levels:
        for b in budgets:
            for model in models:
                for cost in costs:
    
                    expResults = resultsDict[exp][level][b][model]
                    expTrainMask = resultsDict[exp][level][b]['trainMask']
                    expTestMask = resultsDict[exp][level][b]['testMask']
                    labelCost = resultsDict[exp][level]['labellingCosts']
                    groundTruth = resultsDict[exp][level]['yAct']
                    scalerY = resultsDict[exp][level]['scalerY']
                    
                    #Get predicted results
                    
                    yPred = expResults['yPred']
                    
                    predInd = 0
                    predicted = []
                    
                    for i in range(len(expTestMask)):
                        if expTestMask[i]:
                            predicted.append(scalerY.inverse_transform(yPred[predInd].reshape(1, -1))[0])
                            predInd += 1
                        else:
                            predicted.append(scalerY.inverse_transform(groundTruth[i].reshape(1, -1))[0])
                    
                    
                    predictedMean = np.array(predicted)[:,columnDict[cost]['mean']]
                    predictedStdDev = np.array(predicted)[:,columnDict[cost]['std']]
                    
                    if level == 'OD':
                        predictedOMean = []
                        predictedOStdDev = []
                        for o in indexO:
                          
                            indexOfFeatures = indexOD[indexOD['oa_id'] == o].index
                            means= np.array(predictedMean[indexOfFeatures])
                            stdDevs = np.array(predictedStdDev[indexOfFeatures])
                            oWeights = weightsOD['tripCounts'].loc[indexOfFeatures].values
                            predictedOMean.append(np.average(means, weights=oWeights,axis = 0))
                    
                            generatedValues = []
                    
                            for i in range(len(oWeights)):
                                generatedValues = generatedValues + list(np.random.normal(loc=means[i], scale=abs(stdDevs[i]), size=oWeights[i]))
                    
                            predictedOStdDev.append(statistics.stdev(generatedValues))
                    
                            
                            groundTruthO = resultsDict[exp]['O']['yAct']
                            scalerO = resultsDict[exp]['O']['scalerY']
                            
                            actualOMean = scalerO.inverse_transform(groundTruthO)[:,columnDict[cost]['mean']]
                            actualOStd = scalerO.inverse_transform(groundTruthO)[:,columnDict[cost]['std']]
                    else:
                        predictedOMean = predictedMean
                        predictedOStdDev = predictedStdDev

                        actualOMean = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['mean']]
                        actualOStd = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['std']]
                    
                    armPred = np.array(predictedOMean / statistics.mean(predictedOMean))
                    armAct = np.array(actualOMean / statistics.mean(actualOMean))
                    
                    nArmPred = (armPred - armPred.min()) / (armPred.max() - armPred.min())
                    nArmAct = (armAct - armAct.min()) / (armAct.max() - armAct.min())
                    
                    stdrmPred = np.array(np.array(predictedOStdDev) / statistics.mean(predictedOStdDev))
                    stdrmAct = np.array(actualOStd / statistics.mean(actualOStd))
                    
                    nStdrmPred = (stdrmPred - stdrmPred.min()) / (stdrmPred.max() - stdrmPred.min())
                    nStdrmAct = (stdrmAct - stdrmAct.min()) / (stdrmAct.max() - stdrmAct.min())
                    
                    #Perform classification
                    
                    mostlyBad = np.array(indexO)[(nArmPred >= 0.5) & (nStdrmPred >= 0.5)]
                    worst = np.array(indexO)[(nArmPred >= 0.5) & (nStdrmPred < 0.5)]
                    mostlyGood = np.array(indexO)[(nArmPred < 0.5) & (nStdrmPred >= 0.5)]
                    best = np.array(indexO)[(nArmPred < 0.5) & (nStdrmPred < 0.5)]
                    
                    predictedClasses = []
                    
                    for i in indexO:
                        if i in best:
                            predictedClasses.append(1)
                        elif i in mostlyGood:
                            predictedClasses.append(2)
                        elif i in mostlyBad:
                            predictedClasses.append(3)
                        elif i in worst:
                            predictedClasses.append(4)
                        else:
                            print(i)
                    
                    mostlyBad = np.array(indexO)[(nArmAct >= 0.5) & (nStdrmAct >= 0.5)]
                    worst = np.array(indexO)[(nArmAct >= 0.5) & (nStdrmAct < 0.5)]
                    mostlyGood = np.array(indexO)[(nArmAct < 0.5) & (nStdrmAct >= 0.5)]
                    best = np.array(indexO)[(nArmAct < 0.5) & (nStdrmAct < 0.5)]
                    
                    actualClasses = []
                    
                    for i in indexO:
                        if i in best:
                            actualClasses.append(1)
                        elif i in mostlyGood:
                            actualClasses.append(2)
                        elif i in mostlyBad:
                            actualClasses.append(3)
                        elif i in worst:
                            actualClasses.append(4)
                        else:
                            print(i)
                    
                    #For a model get Cost performance (accuracy, MAPE, correlation, PI correlation, classification evaluation)
                    
                    #Compare at o-level
                    #Get yAct
                    #Get yPred
                    #Get error
                    error = abs(actualOMean - predictedOMean)
                    absErrorCost = error.mean()
                    
                    #Get mape
                    errorPct = error / actualOMean.squeeze()
                    absErrorPcntCost = errorPct.mean()
                    
                    jainAct = (actualOMean.sum()**2) / ((actualOMean ** 2).sum()*actualOMean.shape[0])
                    jainPred = (np.array(predictedOMean).sum()**2) / ((np.array(predictedOMean) ** 2).sum()*np.array(predictedOMean).shape[0])
                    
                    jainsErrorCost = abs(jainAct - jainPred)
                    
                    #Get correlation
                    correlationCost = pearsonr(actualOMean,predictedOMean)[0]
                    
                    #std dev performance
                    
                    error = abs(actualOStd - predictedOStdDev)
                    absError = error.mean()
                    
                    #Get mape
                    #errorPct = error / actualOStd.squeeze()
                    #absErrorPcnt = errorPct.mean()
                    
                    #Get correlation
                    correlation = pearsonr(actualOStd,predictedOStdDev)[0]
                    
                    #Classification Scores
                    
                    f1 = f1_score(actualClasses, predictedClasses, average='weighted')
                    recall = recall_score(actualClasses, predictedClasses, average='weighted')
                    precision = precision_score(actualClasses, predictedClasses, average='weighted')
                    accuracy = accuracy_score(actualClasses, predictedClasses)
                    
                    
                    
                    labelSetCost = labelCost[expTrainMask].sum().values[0]
                    fullSetCost = labelCost.sum().values[0]
                    featureExtractTime = resultsDict[exp]['featureExtractionTime']
                    inferenceTimes = statistics.mean(expResults['inference times'])
                    
                    sizeFullMatrix = labelCost.shape[0]
                    sizeLabelMatrix = labelCost[expTrainMask].shape[0]
                    
                    #Get PI
                    
                    for d in demoGroups:
                        population = list(oa_info[d])
                        
                        
                        
                        prm = np.array(np.array(population) / statistics.mean(population))
                        nPrm = (prm - prm.min()) / (prm.max() - prm.min())
                        
                        actualPI = nPrm * nArmAct
                        order = actualPI.argsort()
                        actualPIranks = order.argsort()
                        
                        predPI = nPrm * nArmPred
                        order = predPI.argsort()
                        predPIranks = order.argsort()
                        
                        #Evaluate PI
                        
                        corrPI = pearsonr(actualPI,predPI)[0]
                        corrPIRank = pearsonr(actualPIranks,predPIranks)[0]
                        
                        #System Level
                        
                        #----ACTUAL
                        populationAccessCost = actualOMean * population
                        popACSquared = populationAccessCost**2
                        demoMeanAccess = populationAccessCost.sum()/np.array(population).sum()
                        demoJain = (populationAccessCost.sum() ** 2) / (popACSquared.sum() * popACSquared.shape[0])
                        
                        #----Predicted
                        populationAccessCost = np.array(predictedOMean) * population
                        popACSquared = populationAccessCost**2
                        demoMeanAccessPred = populationAccessCost.sum()/np.array(population).sum()
                        demoJainPred = (populationAccessCost.sum() ** 2) / (popACSquared.sum() * popACSquared.shape[0])
                        
                        #Error
                        demoMeanError = abs(demoMeanAccess - demoMeanAccessPred)
                        demoJainsError = abs(demoJain - demoJainPred)
                        
                        resultsAppend = {}
                        
                        resultsAppend['Experiment'] = exp
                        if paramsDict[exp]['area'] == 'E08000026':
                            resultsAppend['Area'] = 'Cov'
                        else:
                            resultsAppend['Area'] = 'Birm'
                        resultsAppend['POI'] = paramsDict[exp]['poi']
                        resultsAppend['Stratum'] = paramsDict[exp]['stratum']
                        resultsAppend['level'] = level
                        resultsAppend['budget'] = b
                        resultsAppend['model'] = model
                        resultsAppend['cost'] = cost
                        resultsAppend['Demo Group'] = d
                        resultsAppend['Cost Error'] = absErrorCost
                        resultsAppend['Cost MAPE '] = absErrorPcntCost
                        resultsAppend['Jains Error'] = jainsErrorCost
                        resultsAppend['Cost Correlation'] = correlationCost
                        resultsAppend['Std Error'] = absError
                        #resultsAppend['Std MAPE '] = absErrorPcnt
                        resultsAppend['Std Correlation'] = correlation
                        resultsAppend['F1'] = f1
                        resultsAppend['Recall'] = recall
                        resultsAppend['Precision'] = precision
                        resultsAppend['Accuracy'] = accuracy
                        resultsAppend['PI Correlation'] = corrPI
                        resultsAppend['PI Rank Correlation'] = corrPIRank
                        resultsAppend['Demographic Cost Error'] = demoMeanError
                        resultsAppend['Demographic Jains Error '] = demoJainsError
                        resultsAppend['Label Set Cost'] = labelSetCost
                        resultsAppend['Full Label Cost'] = fullSetCost
                        resultsAppend['Feature Extraction Time'] = featureExtractTime
                        resultsAppend['Inference Time'] = inferenceTimes
                        resultsAppend['Size Full Matrix'] = sizeFullMatrix
                        resultsAppend['Size Label Matrix'] = sizeLabelMatrix
                        
                        results.append(resultsAppend)

resultsPD = pd.DataFrame(results)

resultsPD.to_csv('G:/My Drive/University/Working Folder/Transport Access Tool/SSR-Access-Query/results/results_exps_1_to_8.csv')