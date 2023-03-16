import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from methods.OLS import OLSRegression
from methods.MLPRegression import MLPRegression
from methods.GNN import runGNN
from methods.COREG import CoregTrainer
from methods.mean_teacher import *
from methods.label_propogation import runLabelProp
from pipeline.dataPreparation import getTrainingData, getMasks,constructAdjMx
from sklearn.preprocessing import StandardScaler
import time
import sys
import torch

#%% Static Parameters
#TODO: read in as yaml file

if len(sys.argv)>1:
    device = sys.argv[1]
    exp = sys.argv[2]
    # Environment variables
    shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
    oaInfoLoc = 'Data/oa_info.csv'
    labelledTripLoc = 'Data/labelled_sets/'
    unLabelledTripLoc = 'Data/tripSets/'
    dbLoc= 'Data/access.db'
    resultsLoc = 'Data/results/'
    poiLoc = 'Data/pois.csv'
    featuresLoc = 'Data/features/'
    print(device)
    print(exp)

else:
    device = 'cpu'
    exp = 'exp5'
    # Environment variables
    shpFileLoc = 'E:/Data/west_midlands_OAs/west_midlands_OAs.shp'
    oaInfoLoc = 'E:/Data/oa_info.csv'
    labelledTripLoc = 'E:/Data/labelled_sets/'
    unLabelledTripLoc = 'E:/Data/tripSets/'
    dbLoc= 'E:/Data/access.db'
    resultsLoc = 'E:/Data/results/'
    poiLoc = 'E:/Data/pois.csv'
    featuresLoc = 'E:/Data/features/'

#MLP
hiddenMLP1 = 2000
hiddenMLP2 = 1000
hiddenMLP3 = 500
hiddenMLP4 = 250
epochsMLP = 250
dpMLP = 0.2

#GNN
kGNN = 1
dpGNN = 0.2
hiddenGNN1 = 150
hiddenGNN2 = 100
hiddenGNNLinear1 = 75
hiddenGNNLinear2 = 50
epochsGNN = 150

#COREG
numTrialsCoreg = 1
k1Coreg = 1
k2Coreg = 1
p1Coreg = 2
p2Coreg = 2
maxItersCoreg = 25
poolSizeCoreg = 100

#%% Read in exp params

resultsDict = {}

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


params = paramsDict[exp]

boxBuffer = 0.38
intersectionThshHld = 0.15
walkableMins = 6
pcntNbrs = 0.025
vb = 0.3
budgets = [0.025,0.05,0.1,0.3,0.2,0.1,0.07,0.05,0.03]
level = 'O'

#%%
#Read in base datasets
# Read in base data - shapefile / OA info / OA index
wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
oa_info = pd.read_csv(oaInfoLoc)
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#oaLatLons - dictionary associating oa to lat/lons
oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])

poi = pd.read_csv(poiLoc, index_col = 0)
poi = poi[poi['type'] == params['poi']]

#%%

t0 = time.time()
trainingDataDict = getTrainingData(params,poi,unLabelledTripLoc,labelledTripLoc,wm_oas,oaLatLons,walkableMins,oa_info,boxBuffer,False,featuresLoc)
t1 = time.time()

resultsDict[exp] = {
    'featureExtractionTime': t1 - t0
    }

#%%
#for level in levels:

#Standardise data and split into x and y
scalerX = StandardScaler()
x = scalerX.fit_transform(trainingDataDict[level]['features'])
scalerY = StandardScaler()
y = scalerY.fit_transform(trainingDataDict[level]['labels'])

# Create adjacency matrix

edgeIndexNp, edgeWeightsNp = constructAdjMx(trainingDataDict,level,oaLatLons,pcntNbrs)

resultsDict[exp][level] = {
    'scalerX':scalerX,
    'scalerY':scalerY,
    'yAct':y,
    'index':trainingDataDict[level]['index'],
    'weights':trainingDataDict[level]['weights'],
    'labellingCosts':trainingDataDict[level]['labelcosts']
    }

for b in budgets:
    
    print('---- Next Experiment -----')
    print('Level : ' + str(level))
    print('Budget : ' + str(b))
    
    testMask, trainMask, valMask = getMasks(b,vb,x)
    
    resultsDict[exp][level][b] = {
        'testMask':testMask,
        'trainMask':trainMask,
        'valMask':valMask
        }
    
    labeledMask = trainMask + valMask
    unlabeledMask = testMask
    
    print('Starting OLS Regression')
    method = 'OLSRegression'
    
    resultsDict[exp][level][b][method] = {}
    
    yPred = np.zeros(y[testMask].shape)
    infTimes = []
    for i in range(4):
        predVector, infTime = OLSRegression(x,y[:,i],trainMask,testMask)
        yPred[:,i] = predVector
        infTimes.append(infTime)

    resultsDict[exp][level][b][method]['yPred'] = yPred
    resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    print('Starting MLP Regression')
    method = 'MLPRegression'
    
    resultsDict[exp][level][b][method] = {}
    
    yPred = np.zeros(y[testMask].shape)
    infTimes = []
    for i in range(4):
        predVector, infTime, losses, lossesVal = MLPRegression(x,y[:,i],trainMask,testMask,valMask,1000, 500, 250, 100,epochsMLP, device,dpMLP)
        yPred[:,i] = predVector
        infTimes.append(infTime)
        torch.cuda.empty_cache()
    
    resultsDict[exp][level][b][method]['yPred'] = yPred
    resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    # edgeIndexNp = np.load('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeList.npy')
    # edgeWeightsNp = np.load('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeWeight.npy')
    
    print('Starting GNN')
    method = 'GNN'
    resultsDict[exp][level][b][method] = {}
    yPred = np.zeros(y[testMask].shape)
    infTimes = []
    for i in range(4):
        print(i)
        predVector,infTime,losses,lossesVal,lossesTest = runGNN(x,y[:,i],device,edgeIndexNp,edgeWeightsNp,hiddenGNN1,hiddenGNN2,hiddenGNNLinear1,hiddenGNNLinear2,epochsGNN,trainMask,testMask,valMask,kGNN, dpGNN)
        yPred[:,i] = predVector
        infTimes.append(infTime)
        torch.cuda.empty_cache()
        
    resultsDict[exp][level][b][method]['yPred'] = yPred
    resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    print('Starting COREG')
    method = 'COREG'
    
    resultsDict[exp][level][b][method] = {}
    
    yPred = np.zeros(y[testMask].shape)
    infTimes = []
    for i in range(4):
        t0 = time.time()
        coregTrainer = CoregTrainer(numTrialsCoreg,x,y[:,i],trainMask,testMask, k1=k1Coreg, k2=k2Coreg, p1=p1Coreg, p2=p2Coreg, max_iters=maxItersCoreg, pool_size=poolSizeCoreg, verbose=False)
        coregTrainer.run_trials()
        yPred[:,i] = coregTrainer.test_hat.squeeze()
        t1 = time.time()
        infTimes.append(t1 - t0)
    
    resultsDict[exp][level][b][method]['yPred'] = yPred
    resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    print('Starting Mean Teacher')
    method = 'Mean Teacher'
    
    resultsDict[exp][level][b][method] = {}
    
    yPred = np.zeros(y[testMask].shape)
    infTimes = []
    for i in range(4):
        t0 = time.time()
        test_loss, best_val_loss, predVector = optimize_mean_teacher(x,y[:,i],testMask,valMask,labeledMask,unlabeledMask)
        yPred[:,i] = predVector.squeeze()
        t1 = time.time()
        infTimes.append(t1 - t0)
    
    resultsDict[exp][level][b][method]['yPred'] = yPred
    resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    # print('Starting Label Prop')
    # method = 'Label Propogation'
    
    # resultsDict[exp][level][b][method] = {}
    
    # yPred = np.zeros(y[testMask].shape)
    # infTimes = []
    
    # for i in range(4):
    #     print(i)
    #     t0 = time.time()
    #     predVector = runLabelProp(x,y[:,i],testMask,valMask,trainMask,labeledMask,unlabeledMask)
    #     yPred[:,i] = predVector
    #     t1 = time.time()
    #     print(t1 - t0)
    #     infTimes.append(t1 - t0)
    
    # resultsDict[exp][level][b][method]['yAct'] = y
    # resultsDict[exp][level][b][method]['yPred'] = yPred
    # resultsDict[exp][level][b][method]['inference times'] = infTimes
    
    
    f = open(resultsLoc + str(exp) + '.txt', 'wb')
    pickle.dump(resultsDict,f)
    f.close()

#%% For an experiment

# for expNum in allExpParams.keys():
    
#     print('Experiment Number : ' + str(expNum))
    
#     dataDir = 'Data/test_data/'+str(expNum)
#     expParams = allExpParams[expNum]
    
#     # Read in data
    
#     x = np.load(dataDir+'/x.npy')
#     y = np.load(dataDir+'/y.npy')
#     yAct = np.load(dataDir+'/yAct.npy')
#     trainMask = np.load(dataDir+'/trainMask.npy')
#     testMask = np.load(dataDir+'/testMask.npy')
#     valMask = np.load(dataDir+'/valMask.npy')
#     labeledMask = np.load(dataDir+'/labeledMask.npy')
#     unlabeledMask = np.load(dataDir+'/unlabeledMask.npy')
    
#     oaIndex = pd.read_csv(dataDir+'/oa_index.csv', index_col = 0)
#     OPPairs = pd.read_csv(dataDir+'/OPPairs.csv', index_col = 0)
    
#     f = open(dataDir+'/scalerX.txt', 'rb')
#     scalerX = pickle.load(f)
#     f.close()
    
#     f = open(dataDir+'/scalerY.txt', 'rb')
#     scalerY = pickle.load(f)
#     f.close()
    
#     oa_info = oa_info_master.merge(oaIndex, left_on = 'oa_id', right_on = 'oa_id', how='inner')
#     wm_oas = wm_oas_master.merge(oaIndex, left_on = 'OA11CD', right_on = 'oa_id', how='inner')
    
#     #%Run methods
#     #OLS
#     predVector, infTime = OLSRegression(x,y,trainMask,testMask)
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'OLS'
#         modelResults['infTime'] = infTime
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     #MLP
#     predVector, infTime, losses, lossesVal = MLPRegression(x,y,trainMask,testMask,valMask,1000, 500, 250, 100,epochsMLP, device,dpMLP)
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'MLP'
#         modelResults['infTime'] = infTime
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     #GNN - adj mx walk
#     #Load Euclide Matrix
#     edgeIndexNp = np.load('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeList.npy')
#     edgeWeightsNp = np.load('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeWeight.npy')
    
#     predVector,infTime,losses,lossesVal,lossesTest = runGNN(x,y,device,edgeIndexNp,edgeWeightsNp,hiddenGNN1,hiddenGNN2,hiddenGNNLinear1,hiddenGNNLinear2,epochsGNN,trainMask,testMask,valMask,kGNN, dpGNN)
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'GNN - Walk Matrix'
#         modelResults['infTime'] = infTime
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     #GNN - adj mx euclid
#     #TBC - contingent on GNN optimisation
    
#     #GNN - adj mx midway distance
#     #TBC - contingent on ajd mx generation
    
#     #GNN - adj mx midway distance weighted
#     #TBC - contingent on ajd mx generation
    
#     # COREG
#     t0 = time.time()
#     coregTrainer = CoregTrainer(numTrialsCoreg,x,y,trainMask,testMask, k1=k1Coreg,k2=k2Coreg, p1=p1Coreg, p2=p2Coreg, max_iters=maxItersCoreg, pool_size=poolSizeCoreg, verbose=False)
#     coregTrainer.run_trials()
#     predVector = coregTrainer.test_hat
#     t1 = time.time()
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'COREG'
#         modelResults['infTime'] = t1 - t0
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     # Mean Teacher
#     t0 = time.time()
#     test_loss, best_val_loss, predVector = optimize_mean_teacher(x,y,testMask,valMask,labeledMask,unlabeledMask)
#     t1 = time.time()
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'Mean Teacher'
#         modelResults['infTime'] = t1 - t0
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     #Label Propogation
#     t0 = time.time()
#     predVector = runLabelProp(x,y,testMask,valMask,trainMask,labeledMask,unlabeledMask)
#     t1 = time.time()
#     for k,v in dGroups.items():
#         modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,v,wm_oas,mapResults=False)
#         modelResults.update(expParams)
#         modelResults['demo group'] = k
#         modelResults['method'] = 'Label Propogation'
#         modelResults['infTime'] = t1 - t0
#         allResults.append(modelResults)
#     allResultsPD = pd.DataFrame(allResults)
#     allResultsPD.to_csv('results.csv')
    
#     #SSDKL
#     #TBC
