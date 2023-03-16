#Import Modules etc
import pandas as pd
import numpy as np
import pickle
from shapely.geometry import Point
from utils.features import originCenteredFeatures, destinationCenteredFeatured, ODcentredFeatured
import time
from math import radians, cos, sin, asin, sqrt
import random
from sklearn.neighbors import NearestNeighbors

def getOAfromPOI(poiToSearch,poiLonLat,wm_oas,poiInd):
    
    poiIndex = list(poiInd).index(poiToSearch)
    
    point = Point(poiLonLat[poiIndex,0],poiLonLat[poiIndex,1])
    
    for i,r in wm_oas.iterrows():
        if r['geometry'].contains(point):
            oa = r['OA11CD']
            break

    return {'oa':oa, 'lon': poiLonLat[poiIndex,0], 'lat':poiLonLat[poiIndex,1]}


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

# def getTrainingData(p, stratum, targetVar, area, walkableMins, boxBuffer, intersectionThshHld, sr, b, vb, shpFileLoc, oaInfoLoc, dbLoc, urbanCentre, level,k):

#     #Read in base datasets
#     # Read in base data - shapefile / OA info / OA index
#     wm_oas = gpd.read_file(shpFileLoc)
#     wm_oas = wm_oas[wm_oas['LAD11CD'] == area]
#     oa_info = pd.read_csv(oaInfoLoc)
#     oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#     #OA List - list of OA IDs - used for subseuqent SQL query
#     oaList = tuple(list(set(list(oa_info['oa_id']))))
#     oaLats = np.array(oa_info['oa_lat'])
#     oaLons = np.array(oa_info['oa_lon'])
#     cnx = sqlite3.connect(dbLoc)
    
#     #Read in supporting data types for feature sets
    
#     f = open('Data/features/OAsReachableOnFoot.txt', 'rb')
#     OAsReachableOnFoot = pickle.load(f)
#     f.close()
    
#     #Dictionary associating each OA to it's connected OAs (by bus) with informatin about travel time and number of busses in period
#     f = open('Data/features/OAtoOABus.txt', 'rb')
#     OAtoOABus = pickle.load(f)
#     f.close()
    
#     #oaLatLons - dictionary associating oa to lat/lons
#     oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])
    
#     #Identify trips pertaining to OP pairs and get details
    
#     #Select POIs (poiID, lat, lon, type)
#     poi = pd.read_sql_query("SELECT id as poi_id, snapped_longitude as poi_lon, snapped_latitude as poi_lat, type from poi where type = '{}';".format(p),cnx)
    
#     #POI List - list of POI IDs - used for subseuqent SQL query
#     poiList = tuple(list(set(list(poi['poi_id'].astype(str)))))
    
#     #OP Trips - get the trips and calculated results for all OAs and POIs
#     OPTrips = pd.read_sql_query("select b.oa_id,b.poi_id,a.*  from results_full_1 as a inner join (select distinct trip_id, oa_id, poi_id from trips where oa_id in {} and poi_id in {} and stratum = '{}') as b on a.trip_id = b.trip_id".format(oaList,poiList,stratum),cnx)
    
#     #Get unique set of OA and POI ids
#     OPPairs = OPTrips[['oa_id', 'poi_id']].drop_duplicates()
#     OPPairs = OPPairs.reset_index(drop = True)
    
#     #Get index of POIs
#     poiInd = np.array(poi[poi['poi_id'].astype(str).isin(list(set(list(OPPairs['poi_id']))))]['poi_id'])
#     poiLonLat = np.array(poi[poi['poi_id'].astype(str).isin(list(set(list(OPPairs['poi_id']))))][['poi_lon','poi_lat']])
    
#     # Reverse reachability - calculate which OAs are on bus hop away from destination
#     reverseOASearch = {}
#     for i in list(OAtoOABus.keys()):
#         reverseOASearch[i] = []
#     for key, value in OAtoOABus.items():
#         for oa in list(value.keys()):
#             reverseOASearch[oa].append(key)
    
#     #Associate each POI to an OA
#     #Some POIs outside area of study so remove
#     #TODO - this hasn't been optimised at all, need to consider this. KNN search? Precalculate
#     #Associate POIs to OA
#     poiLookUp = {}
#     poisNotFound = []
#     for p in poiInd:
#         try:
#             poiLookUp[p] = getOAfromPOI(p,poiLonLat,wm_oas,poiInd)
#         except:
#             poisNotFound.append(p)
    
#     poiOAId = []
#     dropIndexes = []
#     for i,r in OPPairs.iterrows():
#         try:
#             poiOAId.append(poiLookUp[int(r['poi_id'])]['oa'])
#         except:
#             dropIndexes.append(i)
    
#     # Remove redunadant rows from analysis
#     OPPairs = OPPairs.drop(dropIndexes)
#     OPPairs['poi_oa'] = poiOAId
#     countOAPOI = OPPairs['oa_id'].value_counts()
#     #Only select isntances where OA associates to all POIs
#     #TODO - parameterise this with k
#     OPPairs = OPPairs[OPPairs['oa_id'].isin(countOAPOI[countOAPOI == k].index)]
#     OPTrips = OPTrips.merge(OPPairs,left_on = ['oa_id', 'poi_id'], right_on = ['oa_id', 'poi_id'], how = 'inner')
    
#     #Calculate features for OP set
#     featureVector = []
#     timeChecks = []
#     failed = []
#     count = 0
#     indexSuccess = []
    
#     #calculate a feature set to describe each OA-POI Relationship
#     for i,r in OPPairs.iterrows():
#         count += 1
#         #if count % 10 == 0:
#             #print('Count : ' + str(count))
#         originOA  = r['oa_id']
#         try:
#             t0 = time.time()
#             #destOA = getOAfromPOI(int(r['poi_id']),poiLonLat,wm_oas,poiInd)
#             destOA = r['poi_oa']
#             t1 = time.time()
#             features, times = getOaTOOaFeatVec(oa_info,oaLats,oaLons,oaList,destOA,originOA,boxBuffer,OAsReachableOnFoot,intersectionThshHld,OAtoOABus,reverseOASearch,oaLatLons)
#             times['associate POI'] = t1 - t0
#             featureVector.append(features)
#             timeChecks.append(times)
#             indexSuccess.append(i)
#         except:
#             failed.append({
#                 'index':i,
#                 'o':originOA,
#                 'd':destOA,
#                 'reason':traceback.format_exc()
#                 })
    
#     failed = pd.DataFrame(failed)
#     timeChecks = pd.DataFrame(timeChecks)
#     featureVector = pd.DataFrame(featureVector)
    
#     #Append static features
#     featureVector = staticFeatures(OPPairs.loc[indexSuccess],oa_info,wm_oas,urbanCentre,OAsReachableOnFoot,featureVector)
    
#     #Replace nan values with a bit worse than worse value
#     for var in ['minIntersectionDistD','minIntersectionDistO','meanIntersectionDistD','meanIntersectionDistO']:
#         featureVector[var] = featureVector[var].T.fillna(featureVector[var].max() * 1.333).T
    
#     #Create dictionary for each OA POI combination, value: dataframe of query IDs
#     gbDict = dict(tuple(OPTrips[['oa_id','poi_id','trip_id']].groupby(['oa_id','poi_id'])))
#     #For each key in dictionary randomly select sr% of the IDs
#     queryTripIds = []
#     for k,v in gbDict.items():
#         queryTripIds = queryTripIds + list(v.sample(int(len(v) * sr))['trip_id'])
    
#     # Calculate target variable
#     #calculate actual target variable on trips dataset
#     OPTrips['accessCost'] = (( 1.5 * (OPTrips['total_time'])) - (0.5 * OPTrips['transit_time']) + ((OPTrips['fare'] * 3600) / 6.7) + (10 * OPTrips['num_transfers'])) / 60
#     OPTrips['accessCost']=OPTrips['accessCost'].replace(0,OPTrips['accessCost'].values[np.nonzero(OPTrips['accessCost'].values)].min())
#     OPTrips['total_time']=OPTrips['total_time'].replace(0,OPTrips['total_time'].values[np.nonzero(OPTrips['total_time'].values)].min())
    
#     #OA-POI Level
        
#     if level == 'OA':
    
#         levelGroupBy = ['oa_id']
        
#         #create temp feat vec
#         gpFeatFev = featureVector.copy()
    
#         #Append OA
#         #gpFeatFev['oa_id'] = OPPairs.loc[indexSuccess]['oa_id']
#         gpFeatFev['oa_id'] = list(OPPairs['oa_id'])
#         featureVector = gpFeatFev.groupby('oa_id').mean()
    
#     elif level == 'OAPOI':
    
#         levelGroupBy = ['oa_id', 'poi_id']
    
#     #group on oa-poi for all records (actual)
#     if targetVar == 'Access':
#         targetActual = OPTrips.groupby(levelGroupBy).mean()['accessCost']
#     elif targetVar == 'Time':
#         targetActual = OPTrips.groupby(levelGroupBy).mean()['total_time']
    
#     #group on oa-poi for filtered records (sample)
#     if targetVar == 'Access':
#         targetSample = OPTrips[OPTrips['trip_id'].isin(queryTripIds)].groupby(levelGroupBy).mean()['accessCost']
#     elif targetVar == 'Time':
#         targetSample = OPTrips[OPTrips['trip_id'].isin(queryTripIds)].groupby(levelGroupBy).mean()['total_time']
    
#     #Standardise data and split into x and y
#     scalerX = StandardScaler()
#     x = scalerX.fit_transform(featureVector.values)
#     scalerY = StandardScaler()
#     y = scalerY.fit_transform(targetSample.values.reshape(-1, 1)).squeeze()
#     yAct = targetActual.values.reshape(-1, 1)
    
#     numB = int(x.shape[0] * b)
#     trainRecords = random.sample(range(x.shape[0]), numB)
    
#     numV = int(len(trainRecords) * vb)
#     valRecords = random.sample(trainRecords, numV)
    
#     test = np.full((x.shape[0]), False)
#     train = np.full((x.shape[0]), False)
#     val = np.full((x.shape[0]), False)
    
#     for i in range(x.shape[0]):
#         if i in valRecords:
#             val[i] = True
#         elif i in trainRecords:
#             train[i] = True
#         else:
#             test[i] = True

#     oaMask = oa_info['oa_id'].isin(list(list(OPPairs['oa_id'])))

#     labeledMask = train + val
#     unlabeledMask = test

#     return x, y, yAct, scalerX, scalerY, test, val, train, OPTrips, OPPairs, featureVector, oa_info, wm_oas, oaMask, labeledMask, unlabeledMask

#%%

def getTrainingData(params,poi,unLabelledTripLoc,labelledTripLoc,wm_oas,oaLatLons,walkableMins,oa_info,boxBuffer,saveData,featuresLoc):

    #Read in supporting data types for feature sets
    f = open(featuresLoc+params['OAsReachableFile'], 'rb')
    OAsReachableOnFoot = pickle.load(f)
    f.close()
    
    #Dictionary associating each OA to it's connected OAs (by bus) with informatin about travel time and number of busses in period
    f = open(featuresLoc+params['OAtoOABusFile'], 'rb')
    OAtoOABus = pickle.load(f)
    f.close()
    
    unlabelledTrips = pd.read_csv(unLabelledTripLoc+params['unlabelledTripsFile'])
    labelledTrips = pd.read_csv(labelledTripLoc+params['lablledSetFile'])
    labelledTrips = labelledTrips.merge(unlabelledTrips[['trip_id','oa_id','poi_id']],left_on='trip_id',right_on = 'trip_id', how = 'inner')
    
    #Calculate Generalized Access Cost
    
    labelledTrips['accessCost'] = (( 1.5 * (labelledTrips['total_time'])) - (0.5 * labelledTrips['transit_time']) + ((labelledTrips['fare'] * 3600) / 6.7) + (10 * labelledTrips['num_transfers'])) / 60
    labelledTrips['accessCost']=labelledTrips['accessCost'].replace(0,labelledTrips['accessCost'].values[np.nonzero(labelledTrips['accessCost'].values)].min())
    labelledTrips['total_time']=labelledTrips['total_time'].replace(0,labelledTrips['total_time'].values[np.nonzero(labelledTrips['total_time'].values)].min())

    #Get unique set of OA and POI ids
    ODPairs = labelledTrips[['oa_id', 'poi_id']].drop_duplicates()
    ODPairs = ODPairs.reset_index(drop = True)
        
    #Get index of POIs
    poiInd = np.array(poi[poi['poi_id'].astype(str).isin(list(set(list(ODPairs['poi_id'].astype(str)))))]['poi_id'])
    poiLonLat = np.array(poi[poi['poi_id'].astype(str).isin(list(set(list(ODPairs['poi_id'].astype(str)))))][['poi_lon','poi_lat']])
    
    # Reverse reachability - calculate which OAs are on bus hop away from destination
    reverseOASearch = {}
    for i in list(OAtoOABus.keys()):
        reverseOASearch[i] = []
    for key, value in OAtoOABus.items():
        for oa in list(value.keys()):
            reverseOASearch[oa].append(key)
    
    #Associate each POI to an OA
    #Some POIs outside area of study so remove
    #TODO - this hasn't been optimised at all, need to consider this. KNN search? Precalculate
    #Associate POIs to OA
    poiLookUp = {}
    poisNotFound = []
    for p in poiInd:
        try:
            poiLookUp[p] = getOAfromPOI(p,poiLonLat,wm_oas,poiInd)
        except:
            poisNotFound.append(p)
    
    poiOAId = []
    dropIndexes = []
    for i,r in ODPairs.iterrows():
        try:
            poiOAId.append(poiLookUp[int(r['poi_id'])]['oa'])
        except:
            dropIndexes.append(i)
    
    # Remove redunadant rows from analysis
    ODPairs = ODPairs.drop(dropIndexes)
    ODPairs['poi_oa'] = poiOAId
    
    originsIndex = list(set(list(ODPairs['oa_id'])))
    originsIndex.sort()
    destinationIndexes = list(set(list(ODPairs['poi_oa'])))
    destinationIndexes.sort()

    print('Data Read')
    
    # Origin Centered Features
    t0 = time.time()
    OBTree,hopFeats1,hopFeats2 = originCenteredFeatures(originsIndex,oaLatLons,OAsReachableOnFoot,walkableMins,OAtoOABus,wm_oas,oa_info,ODPairs,params,False,featuresLoc)
    t1 = time.time()
    print('origin level features time')
    print(t1 - t0)
    hopFeats1 = pd.DataFrame(hopFeats1)
    hopFeats2 = pd.DataFrame(hopFeats2)
    
    hopFeats1 = hopFeats1.set_index(['o','d'])
    hopFeats2 = hopFeats2.set_index(['o','d'])
    
    #Destination centerd featured
    
    t0 = time.time()
    IBFeats,IBTree = destinationCenteredFeatured(destinationIndexes,oaLatLons,OAsReachableOnFoot,walkableMins,reverseOASearch,OAtoOABus,wm_oas,oa_info,ODPairs,params)
    t1 = time.time()
    print('destination level features time')
    print(t1 - t0)
    IBFeats = pd.DataFrame(IBFeats)
    IBFeats = IBFeats.set_index(['o','d'])
    
    #OD level featured
    
    t0 = time.time()
    intersectionFeatures = ODcentredFeatured(ODPairs,oaLatLons,OBTree,IBTree,boxBuffer,oa_info,OAsReachableOnFoot)
    t1 = time.time()
    print('OD level features features time')
    print(t1 - t0)
    
    intersectionFeatures = pd.DataFrame(intersectionFeatures)
    intersectionFeatures = intersectionFeatures.set_index(['o','d'])
    
    #Construct feature vector
    
    featVec = ODPairs.merge(intersectionFeatures,left_on = ['oa_id','poi_oa'], right_index = True, how = 'left')
    featVec = featVec.merge(IBFeats,left_on = ['oa_id','poi_oa'], right_index = True, how = 'left')
    featVec = featVec.merge(hopFeats1,left_on = ['oa_id','poi_oa'], right_index = True, how = 'left')
    featVec = featVec.merge(hopFeats2,left_on = ['oa_id','poi_oa'], right_index = True, how = 'left')
    featVec = featVec.drop(columns=['oa_id','poi_oa','poi_id'])
    
    #Data Already on OA Level
    #Get Data on O level with weighted averages
    
    weights = ODPairs[['oa_id', 'poi_id']].merge(labelledTrips[['oa_id', 'poi_id']].value_counts().reset_index(name='tripCounts'),left_on = ['oa_id', 'poi_id'], right_on = ['oa_id', 'poi_id'], how = 'inner')
    
    weightedFeatures = []
    
    for o in originsIndex:
        
        indexOfFeatures = ODPairs[ODPairs['oa_id'] == o].index
        
        feautreToWeigh = featVec.loc[indexOfFeatures].values
        oWeights = weights['tripCounts'].loc[indexOfFeatures].values
        
        weightedFeatures.append(np.average(feautreToWeigh, weights=oWeights,axis = 0))
    
    weightedFeatures = np.array(weightedFeatures)
    
    #Get target and labelling cost on OA Level
    ODLabel = labelledTrips.groupby(['oa_id', 'poi_id']).mean()[['accessCost','total_time']]
    ODLabel = ODLabel.merge(labelledTrips.groupby(['oa_id', 'poi_id']).std()[['accessCost','total_time']],left_index = True, right_index=True, how = 'inner')
    
    print(ODLabel.columns)
    
    #Get target and labelling cost on O Level
    OLabel = labelledTrips.groupby('oa_id').mean()[['accessCost','total_time']]
    OLabel = OLabel.merge(labelledTrips.groupby('oa_id').std()[['accessCost','total_time']],left_index = True, right_index=True, how = 'inner')
    print(OLabel.columns)
    #Get labelling cost
    ODLabelCost = labelledTrips.groupby(['oa_id', 'poi_id']).sum()[['queryTime']]
    OLabelCost = labelledTrips.groupby('oa_id').sum()[['queryTime']]
    
    #Scale
    trainingDataDict = {
        'OD':{'features':featVec.values,
            'labels':ODLabel.values,
            'labelcosts':ODLabelCost,
            'index':ODPairs,
            'weights':weights},
        'O':{'features':weightedFeatures,
             'labels':OLabel.values,
             'labelcosts':OLabelCost,
             'index':originsIndex,
             'weights':None}
            }
    if saveData:
        #Output relevant data
        f = open('C:/Users/chris/My Drive/University/Working Folder/Transport Access Tool/SSR-Access-Query/Data/training_sets/'+params['trainingDataFile'], 'wb')
        pickle.dump(trainingDataDict,f)
        f.close()
    
    print('Training Data Generation Complete.')
    
    return trainingDataDict

#%%

def loadAdj(oaMask, area, level):
    
    if level == 'OA':
        adjMx = np.load('Data/adjMx/' + str(area) + '/adjMx.csv')
        adjMx = adjMx[oaMask]
        adjMx = adjMx[:,oaMask]
    
        edgeList = []
        edgeWeightList = []
        
        for i in range(oaMask.sum()):
            for j in range(oaMask.sum()):
                if adjMx[i,j] > 0:
                    edgeList.append([i,j])
                    edgeWeightList.append(adjMx[i,j])
        
        edgeIndexNp = np.array(edgeList).T
        edgeWeightsNp = np.array(edgeWeightList)
    else:
        pass

    return edgeIndexNp,edgeWeightsNp

#%%

def getMasks(b,vb,x):

    numB = int(x.shape[0] * b)
    trainRecords = random.sample(range(x.shape[0]), numB)
    
    numV = int(len(trainRecords) * vb)
    valRecords = random.sample(trainRecords, numV)
    
    testMask = np.full((x.shape[0]), False)
    trainMask = np.full((x.shape[0]), False)
    valMask = np.full((x.shape[0]), False)
    
    for i in range(x.shape[0]):
        if i in valRecords:
            valMask[i] = True
        elif i in trainRecords:
            trainMask[i] = True
        else:
            testMask[i] = True
        
    return testMask, trainMask, valMask

#%%

def constructAdjMx(trainingDataDict,level,oaLatLons,pcntNbrs):
    
    if level == 'OD':
    #Get lat/lon of all origins in OD index
        odLatLons = np.array(trainingDataDict[level]['index']['oa_id'].map(oaLatLons).to_list())

    else:
        odLatLons = []
        for i in trainingDataDict['O']['index']:
            odLatLons.append(oaLatLons[i])
            
        odLatLons = np.array(odLatLons)

    #Get number of neighbours
    numNeighbours = int(len(odLatLons) * pcntNbrs)

    #train knn
    knn = NearestNeighbors(n_neighbors=numNeighbours)
    knn.fit(odLatLons)

    neighbours = knn.kneighbors(odLatLons, return_distance=True)

    edgeIndexes1 = []
    edgeIndexes2 = []
    edgeWeights = []
    normalized_k = 0.1

    for i in range(len(odLatLons)):
        std = neighbours[0][i].std()
        mxNorm = np.exp(-np.square(neighbours[0][i] / std))
        mxNorm[mxNorm < normalized_k] = 0
        
        edgeIndexes1.append([i] * (mxNorm>0).sum())
        edgeIndexes2.append(list(neighbours[1][0][mxNorm>0]))
        edgeWeights.append(list(mxNorm[mxNorm>0]))

    return np.array([[item for sublist in edgeIndexes1 for item in sublist],[item for sublist in edgeIndexes2 for item in sublist]]), [item for sublist in edgeWeights for item in sublist]