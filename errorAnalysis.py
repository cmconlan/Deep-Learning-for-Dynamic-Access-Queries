import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
import statistics
from scipy.stats.stats import pearsonr
from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

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

columnDict = {'GAC':{
    'mean':0,
    'std':2
    },
    'JT':{
    'mean':1,
    'std':3
        }}

walkableMins = 6

#%%

resultsLoc = 'E:/Data/results/'
featuresLoc = 'E:/Data/features/'
model = 'MLPRegression'
level = 'O'
cost  = 'GAC'

#For an exp
#Get OD index

#%%

allResults = []

exp = 'exp1'


for exp in ['exp1','exp2','exp3','exp4','exp5','exp6','exp7','exp8']:
    b = 0.03
    
    
    params = paramsDict[exp]
    
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    
    oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])
    
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
    
    #Get Origin level error terms (ACSD error corr, MAC error corr)
    
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
    
    
    f = open(featuresLoc+params['OBTreeFile'], 'rb')
    OBTree = pickle.load(f)
    f.close()
    
    #Read in supporting data types for feature sets
    f = open(featuresLoc+params['OAsReachableFile'], 'rb')
    OAsReachableOnFoot = pickle.load(f)
    f.close()
    
    #Dictionary associating each OA to it's connected OAs (by bus) with informatin about travel time and number of busses in period
    f = open(featuresLoc+params['OAtoOABusFile'], 'rb')
    OAtoOABus = pickle.load(f)
    f.close()
    
    reverseOASearch = {}
    for i in list(OAtoOABus.keys()):
        reverseOASearch[i] = []
    for key, value in OAtoOABus.items():
        for oa in list(value.keys()):
            reverseOASearch[oa].append(key)
    
    IBTree = {}
    
    for d in list(set(list(indexOD['poi_oa']))):
        IBTree[d] = {}
        for h in range(1):
            IBTree[d][h+1] = {}
    
        walkableRadiusDest = OAsReachableOnFoot[d][walkableMins]
    
        for a in walkableRadiusDest:
            for t in reverseOASearch[a]:
                bussesToNode = OAtoOABus[t][a]
                IBTree[d][1][t] = [t,statistics.mean(bussesToNode),len(bussesToNode)]
    
    oaPOIFeats = []
    
    count = 0
    
    for i,r in indexOD.iterrows():
        
        count += 1
        
        if count % 100 == 0:
            print(count / indexOD.shape[0])
    
        o = r['oa_id']
        d = r['poi_oa']
        
        originLatLon = oa_info[oa_info['oa_id'] == r['oa_id']][['oa_lat','oa_lon']].values[0]
        destLatLon = oa_info[oa_info['oa_id'] == r['poi_oa']][['oa_lat','oa_lon']].values[0]
        
        destLatLon = oaLatLons[d]
        
        ob = OBTree[o][1]
        ib = IBTree[d][1]
        
        # # Get Intersections
        
        intersectionLatLons = []
        intersections = []
        IBintersections = []
        OBintersections = []
        numImpactIntersectons = 0
        
        # Get OB lat lons
        OBLatLons = []
        for a in ob.keys():
            OBLatLons.append(oaLatLons[a])
        OBLatLons = np.array(OBLatLons)
        
        #Get IB lat lons
        IBLatLons = []
        for a in ib.keys():
            IBLatLons.append(oaLatLons[a])
        IBLatLons = np.array(IBLatLons)
        
        if len(OBLatLons) > 0 and len(IBLatLons) > 0:
        
            #Search nearest neighbours between OB and IB to return list of possible intersection
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(OBLatLons)
            possibleIntersections = knn.kneighbors(IBLatLons, return_distance=False)
        
            #Test if intersections are walkable from one another to determine actual intersections
            countib = -1
        
            #Iterate through all possible iterations
            for i in possibleIntersections:
                countib += 1
                #Possible point on inbound tree
                ibPoint = list(ib.keys())[countib]
                #Possible point on outbound tree
                obPoint = list(ob.keys())[i[0]]
                #get all walkable OA witin x mins of OB points
                walkablePointFromOB = OAsReachableOnFoot[obPoint][8]
                #Test if Inbound point is walkable
                if ibPoint in walkablePointFromOB:
                    IBintersections.append(ib[ibPoint])
                    OBintersections.append(ob[obPoint])
                    intersectionLatLons.append(oaLatLons[ibPoint])
                    intersections.append([obPoint,ibPoint])
                    
            intersectionLatLons = np.array(intersectionLatLons)
        
        euclidDist = haversine(originLatLon[0], originLatLon[1], destLatLon[0], destLatLon[1])
        numIBLeafs = len(ib)
        numOBLeafs = len(ob)
        numIntersections = len(intersections)
        
        oaPOIFeats.append({
            'oa_id':o,
            'poi_id':r['poi_id'],
            'poi_oa':d,
            'dist':euclidDist,
            'numIBLeafs':numIBLeafs,
            'numOBLeafs':numOBLeafs,
            'numIntersections':numIntersections
            })
    
    oaPOIFeatsPD = pd.DataFrame(oaPOIFeats)
    
    #Append weights
    
    oaPOIFeatsPD = oaPOIFeatsPD.merge(weightsOD, left_on = ['oa_id','poi_id'], right_on = ['oa_id','poi_id'], how = 'inner')
    
    # Weighted mean
    
    indxCount = 0
    for o in indexO:
        print(indxCount)
        featureToWeigh = oaPOIFeatsPD[oaPOIFeatsPD['oa_id'] == o][['dist','numOBLeafs','numIBLeafs','numIntersections']]
        weights = oaPOIFeatsPD[oaPOIFeatsPD['oa_id'] == o]['tripCounts']
        weightedFeatures = np.average(featureToWeigh, weights=weights,axis = 0)
        allResults.append({
            'city': params['area'],
            'poiType': params['poi'],
            'MAC Error': abs(predictedOMean[indxCount]-actualOMean[indxCount]),
            'STD Error': abs(predictedOStdDev[indxCount]-actualOStd[indxCount]),
            'Dist' : weightedFeatures[0] ,
            'numOBLeafs': weightedFeatures[1],
            'numIBLeafs': weightedFeatures[2],
            'numIntersections':weightedFeatures[3]
            })
        indxCount += 1

#%%

allResultsPD = pd.DataFrame(allResults)

#%%
allResultsPD.to_csv('G:/My Drive/University/Working Folder/Transport Access Tool/SSR-Access-Query/errorAnalysis.csv')


#%%

allResultsPD = pd.read_csv('G:/My Drive/University/Working Folder/Transport Access Tool/SSR-Access-Query/errorAnalysis.csv', index_col = 0)

#%%

#Birm - E08000025
#Cov - E08000026

birm = allResultsPD[allResultsPD['city'] == 'E08000025']
cov = allResultsPD[allResultsPD['city'] == 'E08000026']

#%%

predictors = ['Dist', 'numOBLeafs','numIBLeafs', 'numIntersections']

for p in predictors:

    target = allResultsPD['MAC Error'].values
    predictor = allResultsPD[p].values
    print(p)
    print(pearsonr(target,predictor)[0])

#%%

birm['Dist'].mean()

#%% Compare error rate to actual std dev

allResults = []

exp = 'exp1'
b = 0.03

for exp in ['exp1','exp2','exp3','exp4','exp5','exp6','exp7','exp8']:
    
    params = paramsDict[exp]
    
    wm_oas = gpd.read_file(shpFileLoc)
    wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
    oa_info = pd.read_csv(oaInfoLoc)
    oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
    
    oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])
    
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
    
    #Get Origin level error terms (ACSD error corr, MAC error corr)
    
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
    
    predictedOMean = predictedMean
    predictedOStdDev = predictedStdDev
    
    actualOMean = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['mean']]
    actualOStd = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['std']]
    
    errorsHigh = []
    errorLow = []
    
    actualsLow = []
    actualsHigh = []
    predLow = []
    predHigh = []
    
    indxCount = 0
    
    for i in actualOStd:
        if i == 0:
            errorLow.append(abs(predictedOStdDev[indxCount] - actualOStd[indxCount]))
            actualsLow.append(actualOStd[indxCount])
            predLow.append(predictedOStdDev[indxCount])
        else:
            errorsHigh.append(abs(predictedOStdDev[indxCount] - actualOStd[indxCount]))
            actualsHigh.append(actualOStd[indxCount])
            predHigh.append(predictedOStdDev[indxCount])
        indxCount += 1
    
    print(np.array(errorLow).mean())
    print(np.array(errorsHigh).mean())
    
    print((len(errorLow)/len(actualOStd))*100)
    labelTrips = pd.read_csv('E:/Data/labelled_sets/'+params['lablledSetFile'])
    
    if params['poi'] != 'Job Centre':
    
        allResults.append({
            'city': params['area'],
            'poiType': params['poi'],
            'lowStdError':np.array(errorLow).mean(),
            'highStdError':np.array(errorsHigh).mean(),
            'lowCorr':pearsonr(actualsLow,predLow)[0],
            'highCorr':pearsonr(actualsHigh,predHigh)[0],
            'percentageLowStg':(len(errorLow)/len(actualOStd))*100,
            'percentWalkTrips':(labelTrips['transit_time'] == 0).sum()/labelTrips['transit_time'].shape[0],
            'numWalkingTrips':(labelTrips['transit_time'] == 0).sum(),
            'numTrips':labelTrips.shape[0],
            'avgOAsAssociated':indexOD['oa_id'].value_counts(0).mean()
            })

allResultsPD = pd.DataFrame(allResults)

#%%

actualsLow

#%%
exp = 'exp5'

params = paramsDict[exp]

wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
oa_info = pd.read_csv(oaInfoLoc)
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')

oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])

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

#Get Origin level error terms (ACSD error corr, MAC error corr)

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

predictedOMean = predictedMean
predictedOStdDev = predictedStdDev

actualOMean = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['mean']]
actualOStd = scalerY.inverse_transform(groundTruth)[:,columnDict[cost]['std']]

errorsHigh = []
errorLow = []

actualsLow = []
actualsHigh = []
predLow = []
predHigh = []

indxCount = 0

for i in actualOStd:
    if i == 0:
        errorLow.append(abs(predictedOStdDev[indxCount] - actualOStd[indxCount]))
        actualsLow.append(actualOStd[indxCount])
        predLow.append(predictedOStdDev[indxCount])
    else:
        errorsHigh.append(abs(predictedOStdDev[indxCount] - actualOStd[indxCount]))
        actualsHigh.append(actualOStd[indxCount])
        predHigh.append(predictedOStdDev[indxCount])
    indxCount += 1

#%%


#%%

#%%

#%%

#%%


#%%

#%%