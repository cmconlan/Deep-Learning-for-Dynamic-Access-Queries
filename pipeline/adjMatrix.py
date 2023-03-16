#%% --------------- DEVELOPMENT OF ADJ MX
#Create ADJ MX
#Create link when there is an association between the origin points

import numpy as np
import pickle
from math import radians, cos, sin, asin, sqrt

#Function to get euclidean distance
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

#Get filtered version of oa info master to create oaMask
_temp = oa_info_master.merge(wm_oas_master[wm_oas_master['LAD11CD'] == expParams['area']][['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')

#Create OA Mask
oaMask = _temp['oa_id'].isin(list(list(OPPairs['oa_id'])))

#Read in Euclid Matrix
euclidMx = np.load('Data/adjMx/' + str(expParams['area']) + '/euclidMx.csv')

#Filter on OA mask
euclidMx = euclidMx[oaMask]
euclidMx = euclidMx[:,oaMask]

adjMx = np.zeros(euclidMx.shape)

#%%
# Get walking radius'
f = open('Data/features/OAsReachableOnFoot.txt', 'rb')
OAsReachableOnFoot = pickle.load(f)
f.close()

#%%
# For each OA get OAs with x mins walking radius to form associations
#Iterate through each OA
oaIndexFilter = list(oaIndex[oaMask]['oa_id'])
for k,v in OAsReachableOnFoot.items():
    #All OA walkable in x minutes (e.g., 20)
    nodesAssociations = v[20]
    #Get index of node
    try:
        baseNode = oaIndexFilter.index(k)
    except:
        continue
    #Create ditance and association matrix
    distances = []
    associationIndex = []
    #For each associated node
    for node in nodesAssociations:
        try:
            associatedNode = oaIndexFilter.index(node)
        except:
            continue
        distances.append(euclidMx[baseNode,associatedNode])
        associationIndex.append(associatedNode)
    distances = np.array(distances)
    #Noralize distance matrix
    normalisedDist = 1 - (distances - distances.min()) / (distances.max() - distances.min())
    
    for i in range(len(normalisedDist)):
        adjMx[baseNode,associationIndex[i]] = normalisedDist[i]

#%% o-walk-matrix

#Create origin index
#Adjacency matrix entry for the origin point of each O-P query
originIndex = []
for i in list(OPPairs['oa_id']):
    originIndex.append(oaIndexFilter.index(i))

#Create adjacency matrix for GNN input where each OP query is associated to each other OP query by the distance given between their origin points
edgeList = []
edgeWeightList = []

count = 0

iterations = len(OPPairs) * len(OPPairs)

iIndex = -1
for i in originIndex:
    iIndex += 1
    jIndex = -1
    for j in originIndex:
        jIndex += 1
        count += 1
        if count % 1000 == 0:
            print((count / iterations)*100)
        if i == j:
            edgeList.append([iIndex,jIndex])
            edgeWeightList.append(1)
        elif adjMx[i,j] > 0:
            edgeList.append([iIndex,jIndex])
            edgeWeightList.append(adjMx[i,j])

edgeIndexNp = np.array(edgeList).T
edgeWeightsNp = np.array(edgeWeightList)

np.save('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeList.npy', edgeIndexNp)
np.save('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-walk-edgeWeight.npy', edgeWeightsNp)

#%% d-walk-matrix


#%% o eucliden with gaussian exponential normalising

#Gaussian kernel normalization
normalized_k = 0.4
euclidMxFlat = euclidMx[~np.isinf(euclidMx)].flatten()
std = euclidMxFlat.std()
euclidMxNorm = np.exp(-np.square(euclidMx / std))
euclidMxNorm[euclidMxNorm < normalized_k] = 0
euclidMxNorm[euclidMxNorm >= 1] = 0

# print(np.count_nonzero(euclidMxNorm[0])/euclidMxNorm.shape[0])
# _ = plt.hist(euclidMxNorm[0], bins='auto')

#Create adjacency matrix for GNN input where each OP query is associated to each other OP query by the distance given between their origin points
edgeList = []
edgeWeightList = []

count = 0

iterations = len(OPPairs) * len(OPPairs)

iIndex = -1
for i in originIndex:
    iIndex += 1
    jIndex = -1
    for j in originIndex:
        jIndex += 1
        count += 1
        if count % 1000 == 0:
            print((count / iterations)*100)
        # if i == j:
        #     edgeList.append([iIndex,jIndex])
        #     edgeWeightList.append(1)
        mxLkUp = euclidMxNorm[i,j]
        if mxLkUp > 0:
            edgeList.append([iIndex,jIndex])
            edgeWeightList.append(mxLkUp)

edgeIndexNp = np.array(edgeList).T
edgeWeightsNp = np.array(edgeWeightList)

np.save('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-euclid-edgeList.npy', edgeIndexNp)
np.save('Data/adjMx/' + str(expParams['area']) + '/'+str(expParams['p'])+'-o-euclid-edgeWeight.npy', edgeWeightsNp)

#%% euclid distance between query midpoints

for each query, get the midpoint:
    
calculate the pairwise euclidean mid points

perform gaussian normalization

#%%
#Create midpoint mx
midPntMx = np.zeros((OPPairs.shape[0],OPPairs.shape[0]))
#Create midpoint mx
midPntWeightedMx = np.zeros((OPPairs.shape[0],OPPairs.shape[0]))

index1 = -1

totalIterations = OPPairs.shape[0] * OPPairs.shape[0]

iterations = 0

import time

t0 = time.time()

for i1,r1 in OPPairs[:150].iterrows():
    originLat1 = oa_info[oa_info['oa_id'] == r1['oa_id']]['oa_lat'].values[0]
    originLon1 = oa_info[oa_info['oa_id'] == r1['oa_id']]['oa_lon'].values[0]
    destLat1 = oa_info[oa_info['oa_id'] == r1['poi_oa']]['oa_lat'].values[0]
    destLon1 = oa_info[oa_info['oa_id'] == r1['poi_oa']]['oa_lon'].values[0]

    #Get midway points

    midwayPointLat1 = (originLat1 + destLat1) / 2
    midwayPointLon1 = (originLon1 + destLon1) / 2

    quarterMidPointLat1 = (originLat1 + midwayPointLat1) / 2
    quarterMidPointLon1 = (originLon1 + midwayPointLon1) / 2

    threeQuerterMidPointLat1 = (destLat1 + midwayPointLat1) / 2
    threeQuerterMidPointLon1 = (destLon1 + midwayPointLon1) / 2
    index1 += 1
    index2 = -1
    for i2,r2 in OPPairs[:75].iterrows():
        
        iterations += 1
        
        if iterations % 1000 == 0:
            print(str((iterations / totalIterations)*100))
        
        originLat2 = oa_info[oa_info['oa_id'] == r2['oa_id']]['oa_lat'].values[0]
        originLon2 = oa_info[oa_info['oa_id'] == r2['oa_id']]['oa_lon'].values[0]
        destLat2 = oa_info[oa_info['oa_id'] == r2['poi_oa']]['oa_lat'].values[0]
        destLon2 = oa_info[oa_info['oa_id'] == r2['poi_oa']]['oa_lon'].values[0]

        midwayPointLat2 = (originLat2 + destLat2) / 2
        midwayPointLon2 = (originLon2 + destLon2) / 2

        quarterMidPointLat2 = (originLat2 + midwayPointLat2) / 2
        quarterMidPointLon2 = (originLon2 + midwayPointLon2) / 2

        threeQuerterMidPointLat2 = (destLat2 + midwayPointLat2) / 2
        threeQuerterMidPointLon2 = (destLon2 + midwayPointLon2) / 2
        
        midPointDist = haversine(midwayPointLon1, midwayPointLat1, midwayPointLon2, midwayPointLat2)

        #Origin match
        if r1['oa_id'] == r2['oa_id']:
            distWeighted = haversine(quarterMidPointLon1, quarterMidPointLat1, quarterMidPointLon2, quarterMidPointLat2)
        elif r1['poi_oa'] == r2['poi_oa']:
            distWeighted = haversine(threeQuerterMidPointLon1, threeQuerterMidPointLat1, threeQuerterMidPointLon2, threeQuerterMidPointLat2)
        else:
            distWeighted = midPointDist
        
        index2 += 1
        
        midPntMx[index1,index2] = midPointDist
        midPntWeightedMx[index1,index2] = distWeighted

t1 = time.time()
print('Time Taken : ' + str(t1 - t0))
      
#%%



plt.scatter(originLat1,originLon1, color = 'green')
plt.scatter(destLat1,destLon1, color = 'red')
plt.scatter(midwayPointLat1,midwayPointLon1)
plt.scatter(quarterMidPointLat1,quarterMidPointLon1)
plt.scatter(threeQuerterMidPointLat1,threeQuerterMidPointLon1)
plt.show()

#%%



plt.scatter(originLat2,originLon2, color = 'green')
plt.scatter(destLat2,destLon2, color = 'red')
plt.scatter(midwayPointLat2,midwayPointLon2)
plt.scatter(quarterMidPointLat2,quarterMidPointLon2)
plt.scatter(threeQuerterMidPointLat2,threeQuerterMidPointLon2)
plt.show()

#%%



#%%



#If origins match
#Take midway point at 25% of euclid distance for both queries

#If distinations match
#Take midway point at 75% of euclid distance for both queries

#If no matching
#Take midway pont at 50%

#else
#mid way point at 50%

#%% Calculate euclid distance between midway points




#%%

import pandas as pd
import math

validateAdj = pd.DataFrame(index = range(x.shape[0]), columns = ['associated nodes','number edges', 'edge weights'])
validateAdj['number edges'] = 0

associatedEdge = []
edgeWeights = []
lastEdge = edgeList[0][0]
newEdge = False

count = -1
iterations = len(edgeList)

for i in edgeList:
    count += 1
    if count % 500 == 0:
        print((count/iterations)*100)
    validateAdj.loc[i[0],'number edges'] += 1
    associatedEdge.append(i[1])
    edgeWeights.append(edgeWeightList[count])
    validateAdj.at[i[0],'associated nodes'] = associatedEdge
    validateAdj.at[i[0],'edge weights'] = edgeWeights
    
    if i[0] != lastEdge:
        newEdge = True
    
    lastEdge = i[0]
    
    if newEdge:
        associatedEdge = []
        edgeWeights = []
        newEdge = False

debug = validateAdj.loc[5337]

#%%
hist = validateAdj['number edges'].hist(bins=20)
print(validateAdj['number edges'].min())
print(validateAdj['number edges'].max())

#%%

randomSample = validateAdj.sample(1)
print(randomSample.index[0])
_ = plt.hist(list(randomSample['edge weights']), bins='auto', alpha = 0.35)

#%%

for i in range(50):
    randomSample = validateAdj.sample(1)
    _ = plt.hist(list(randomSample['edge weights']), bins='auto', alpha = 0.35)

#%% Map some instances

#Randomly select instance
randomSample = validateAdj.sample(1)
baseOa = oaIndex[originIndex[randomSample.index[0]]]

associatedNodes = []

for i in range(len(randomSample['associated nodes'].values[0])):
    appendRow = {
        'oa':oaIndex[originIndex[list(randomSample['associated nodes'])[0][i]]],
        'weight':list(randomSample['edge weights'])[0][i]}
    associatedNodes.append(appendRow)
associatedNodes = pd.DataFrame(associatedNodes)
associatedNodes = associatedNodes.groupby('oa').mean()


#Filter wm-oas on list

plotOAs = wm_oas.merge(associatedNodes,left_on = 'OA11CD', right_on = 'oa', how = 'inner')

colours = []
for i,r in wm_oas.iterrows():

    if r['OA11CD'] in list(associatedNodes.index):
        colours.append('red')
    else:
        colours.append('blue')

plotOAs.plot(column='weight', cmap='OrRd', scheme='percentiles')
plt.scatter(oa_info[oa_info['oa_id'] == baseOa]['oa_lon'],oa_info[oa_info['oa_id'] == baseOa]['oa_lat'],s = 20)
plt.show()

#Attach the associations given in adjacency matrix

wm_oas.plot(color = colours)
#plt.scatter(oa_info[oa_info['oa_id'] == baseOa]['oa_lon'],oa_info[oa_info['oa_id'] == baseOa]['oa_lat'],s = 1)
plt.show()

