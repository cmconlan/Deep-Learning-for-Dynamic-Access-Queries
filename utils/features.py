from math import radians, cos, sin, asin, sqrt
#from shapely.geometry import Point, LineString, Polygon
import shapely
import statistics
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
from operator import itemgetter
import pickle
#%%

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

def forwardHops(OAsReachableOnFoot,originOA,destOA,originLat,originLon,destLat,destLon,OAinBbox,oaLatLons,OAtoOABus,walkableMins=6,numHops = 2):
    forwardHopDict = {}
    for h in range(numHops):
        forwardHopDict[h] = {}
        forwardHopDict[h]['reachableNodes'] = []
        forwardHopDict[h]['transitNodes'] = []
        forwardHopDict[h]['destReachable'] = False
        forwardHopDict[h]['reachalbeNodesInBox'] = []
        forwardHopDict[h]['distToDest'] = haversine(originLat,originLon,destLat,destLon)
        forwardHopDict[h]['distToOrigin'] = 0
        forwardHopDict[h]['transitTimeToNearestNode'] = 0
        forwardHopDict[h]['numBussesToNearestNode'] = 0
    
    # walkableRadiusOrigin = oasReachableNMin[originOA]
    # if originOA not in walkableRadiusOrigin:
    #     walkableRadiusOrigin.append(originOA)
    
    walkableRadiusOrigin = OAsReachableOnFoot[originOA][walkableMins]
    
    for o in walkableRadiusOrigin:
        h = 0
        #calculate o features    
        #Update Feature Dict
        forwardHopDict[h]['reachableNodes'].append(o)
        if o == destOA:
            forwardHopDict[h]['destReachable'] = True
        #Test dist to destination, if lower than existing update
        distToDest = haversine(oaLatLons[o][0],oaLatLons[o][1],destLat,destLon)
        if distToDest < forwardHopDict[h]['distToDest']:
            forwardHopDict[h]['distToDest'] = distToDest
            forwardHopDict[h]['distToOrigin'] = haversine(oaLatLons[o][0],oaLatLons[o][1], originLat, originLon)
            forwardHopDict[h]['transitTimeToNearestNode'] = 0
            forwardHopDict[h]['numBussesToNearestNode'] = 0
        if o in OAinBbox:
            forwardHopDict[h]['reachalbeNodesInBox'].append(o)
        
        for t in dict.fromkeys(list(OAtoOABus[o].keys()), o):
            #Update Feature Dict
            forwardHopDict[h]['reachableNodes'].append(t)
            forwardHopDict[h]['transitNodes'].append(t)
            if t == destOA:
                forwardHopDict[h]['destReachable'] = True
            #Test dist to destination, if lower than existing update
            distToDest = haversine(oaLatLons[t][0],oaLatLons[t][1],destLat,destLon)
            if distToDest < forwardHopDict[h]['distToDest']:
                forwardHopDict[h]['distToDest'] = distToDest
                forwardHopDict[h]['distToOrigin'] = haversine(oaLatLons[t][0],oaLatLons[t][1], originLat, originLon)
                bussesToNode = OAtoOABus[o][t]
                forwardHopDict[h]['transitTimeToNearestNode'] = statistics.mean(bussesToNode)
                forwardHopDict[h]['numBussesToNearestNode'] = len(bussesToNode)
            if t in OAinBbox:
                forwardHopDict[h]['reachalbeNodesInBox'].append(t)
            
            h += 1
            for t_2 in dict.fromkeys(list(OAtoOABus[t].keys()), o):
                forwardHopDict[h]['reachableNodes'].append(t_2)
                forwardHopDict[h]['transitNodes'].append(t_2)
                if t_2 == destOA:
                    forwardHopDict[h]['destReachable'] = True
                distToDest = haversine(oaLatLons[t_2][0],oaLatLons[t_2][1],destLat,destLon)
                if distToDest < forwardHopDict[h]['distToDest']:
                    forwardHopDict[h]['distToDest'] = distToDest
                    forwardHopDict[h]['distToOrigin'] = haversine(oaLatLons[t_2][0],oaLatLons[t_2][1], originLat, originLon)
                    bussesToNode = OAtoOABus[t][t_2]
                    forwardHopDict[h]['transitTimeToNearestNode'] = statistics.mean(bussesToNode)
                    forwardHopDict[h]['numBussesToNearestNode'] = len(bussesToNode)
                if t_2 in OAinBbox:
                    forwardHopDict[h]['reachalbeNodesInBox'].append(t_2)
            h -= 1

    return forwardHopDict,walkableRadiusOrigin

def reverseHop(OAsReachableOnFoot,originOA,destOA,originLat,originLon,destLat,destLon,OAinBbox,oaLatLons,OAtoOABus,reverseOASearch,walkableMins=6):

    reverseHopDict = {}
    reverseHopDict['reachableNodes'] = []
    reverseHopDict['destReachable'] = False
    reverseHopDict['reachalbeNodesInBox'] = []
    reverseHopDict['distToDest'] = 0
    reverseHopDict['distToOrigin'] = haversine(originLat,originLon,destLat,destLon)
    reverseHopDict['transitTimeToNearestNode'] = 0
    reverseHopDict['numBussesToNearestNode'] = 0
    
    #Get walkable OAs from dest
    #From each walkable OA, get buses and add features to reverse hop dict
    
    #Get list of OAs walkable from point of destination
    # walkableRadiusDest = oasReachableNMin[destOA]
    # if destOA not in walkableRadiusDest:
    #     walkableRadiusDest.append(destOA)
    
    walkableRadiusDest = OAsReachableOnFoot[destOA][walkableMins]
    
    for o in walkableRadiusDest:
        for t in reverseOASearch[o]:
            reverseHopDict['reachableNodes'].append(t)
            if t == originOA:
                reverseHopDict['destReachable'] = True
            distToOrigin = haversine(oaLatLons[t][0],oaLatLons[t][1],originLat,originLon)
            if distToOrigin < reverseHopDict['distToOrigin']:
                reverseHopDict['distToDest'] = haversine(oaLatLons[t][0],oaLatLons[t][1],destLat,destLon)
                reverseHopDict['distToOrigin'] = distToOrigin
                bussesToNode = OAtoOABus[t][o]
                reverseHopDict['transitTimeToNearestNode'] = statistics.mean(bussesToNode)
                reverseHopDict['numBussesToNearestNode'] = len(bussesToNode)
            if t in OAinBbox:
                reverseHopDict['reachalbeNodesInBox'].append(t)
    return reverseHopDict, walkableRadiusDest
# Definition - getBbox
#input - originLat,destLat, originLon,destLon, boxBuffer
#return - bbox

def getBbox(originLat,destLat, originLon,destLon, boxBufferLon, boxBufferLat):
    latitudes = [originLat,destLat]
    longitudes = [originLon,destLon]
        
    latDist = max(latitudes) - min(latitudes)
    latBuffer = latDist * boxBufferLat
    lonDist = max(longitudes) - min(longitudes)
    lonBuffer = lonDist * boxBufferLon
    
    minLat = min(latitudes) - latBuffer
    maxLat = max(latitudes) + latBuffer
    minLon = min(longitudes) - lonBuffer
    maxLon = max(longitudes) + lonBuffer
    
    return shapely.geometry.box(*(minLon,minLat,maxLon,maxLat), ccw=True)



def getBoundingBoxOAs(originLat,originLon,destLat, destLon,boxBuffer,oa_info):
    
    #Expand min/max obserbed lats/lons by some buffer
    latitudes = [originLat,destLat]
    longitudes = [originLon,destLon]
        
    latDist = max(latitudes) - min(latitudes)
    latBuffer = latDist * boxBuffer
    lonDist = max(longitudes) - min(longitudes)
    lonBuffer = lonDist * boxBuffer
    
    minLat = min(latitudes) - latBuffer
    maxLat = max(latitudes) + latBuffer
    minLon = min(longitudes) - lonBuffer
    maxLon = max(longitudes) + lonBuffer

    #Output OAs within box
    return list(oa_info['oa_id'][(oa_info['oa_lat'] > minLat) & (oa_info['oa_lat'] < maxLat) & (oa_info['oa_lon'] > minLon) * (oa_info['oa_lon'] < maxLon)])

def getIntersections(o,d,OBTree,IBTree,oaLatLons,bBox,OAsReachableOnFoot):
    
    intersectionLatLons = []
    intersections = []
    IBintersections = []
    OBintersections = []
    numImpactIntersectons = 0
    #Load caches trees
    ob = OBTree[o]
    ib = IBTree[d]

    # Get OB lat lons
    OBLatLons = []
    for a in ob[1].keys():
        OBLatLons.append(oaLatLons[a])
    OBLatLons = np.array(OBLatLons)

    #Get IB lat lons
    IBLatLons = []
    for a in ib[1].keys():
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
            ibPoint = list(ib[1].keys())[countib]
            #Possible point on outbound tree
            obPoint = list(ob[1].keys())[i[0]]
            #get all walkable OA witin x mins of OB points
            walkablePointFromOB = OAsReachableOnFoot[obPoint][8]
            #Test if Inbound point is walkable
            if ibPoint in walkablePointFromOB:
                IBintersections.append(ib[1][ibPoint])
                OBintersections.append(ob[1][obPoint])
                intersectionLatLons.append(oaLatLons[obPoint])
                intersections.append([obPoint,ibPoint])
                if (ibPoint in bBox) or (obPoint in bBox):
                    numImpactIntersectons += 1
                
        intersectionLatLons = np.array(intersectionLatLons)

    return intersectionLatLons,intersections,IBintersections,OBintersections,OBLatLons,IBLatLons, numImpactIntersectons

#%%

def getOaTOOaFeatVec(oa_info,oaLats,oaLons,oaList,destOA,originOA,boxBuffer,OAsReachableOnFoot,threshold,OAtoOABus,reverseOASearch,oaLatLons):

    featDict = {}
    timeChecks = {}
    
    #Calculate a bouncing box around the origin and destinaton points, and select the intersecting OAs
    t0 = time.time()
    destLat = oa_info[oa_info['oa_id'] == destOA]['oa_lat'].values[0]
    destLon = oa_info[oa_info['oa_id'] == destOA]['oa_lon'].values[0]
    
    originLat = oa_info[oa_info['oa_id'] == originOA]['oa_lat'].values[0]
    originLon = oa_info[oa_info['oa_id'] == originOA]['oa_lon'].values[0]
    
    OAinBbox = getBoundingBoxOAs(originLat,originLon,destLat, destLon,boxBuffer,oaLats,oaLons,oa_info)
    t1 = time.time()
    timeChecks['Bounding Box Interactions'] = t1 - t0
    
    #forwardHop
    t0 = time.time()
    forwardHopDict, walkableRadiusOrigin = forwardHops(OAsReachableOnFoot,originOA,destOA,originLat,originLon,destLat,destLon,OAinBbox,oaLatLons,OAtoOABus,walkableMins=6,numHops = 2)     
    t1 = time.time()
    timeChecks['Forward Hops'] = t1 - t0
    
    t0 = time.time()          
    reverseHopDict, walkableRadiusDest = reverseHop(OAsReachableOnFoot,originOA,destOA,originLat,originLon,destLat,destLon,OAinBbox,oaLatLons,OAtoOABus,reverseOASearch,walkableMins=6)
    t1 = time.time()
    timeChecks['Reverse Hops'] = t1 - t0
    
    #Add Featrues to feature vector
    
    t0 = time.time()
    featDict['nodesReachable1Hop'] = len(set(forwardHopDict[0]['reachableNodes'])) / len(oaList)
    featDict['nodesReachable2Hops'] = len(set(forwardHopDict[0]['reachableNodes'] + forwardHopDict[1]['reachableNodes'])) / len(oaList)
    featDict['nodesReachableReverse'] = len(set(reverseHopDict['reachableNodes'])) / len(oaList)
    
    featDict['nodesReachable1HopBox'] = len(set(forwardHopDict[0]['reachalbeNodesInBox'])) / len(oaList)
    featDict['nodesReachable2HopsBox'] = len(set(forwardHopDict[0]['reachalbeNodesInBox'] + forwardHopDict[1]['reachalbeNodesInBox'])) / len(oaList)
    featDict['nodesReachableReverseBox'] = len(set(reverseHopDict['reachalbeNodesInBox'])) / len(oaList)
    
    featDict['destReachable1Hop'] = forwardHopDict[0]['destReachable']
    featDict['destReachable2Hop'] = forwardHopDict[1]['destReachable']
    featDict['destReachableReverse'] = reverseHopDict['destReachable']
    
    featDict['distToDest1Hop'] = forwardHopDict[0]['distToDest']
    featDict['distToDest2Hop'] = forwardHopDict[1]['distToDest']
    featDict['distToDestReverse'] = reverseHopDict['distToDest']
    
    featDict['distToOrigin1Hop'] = forwardHopDict[0]['distToOrigin']
    featDict['distToOrigin2Hop'] = forwardHopDict[1]['distToOrigin']
    featDict['distToOriginReverse'] = reverseHopDict['distToOrigin']
    
    featDict['transitTimeNearestNode1Hop'] = forwardHopDict[0]['transitTimeToNearestNode']
    featDict['transitTimeNearestNode2Hop'] = forwardHopDict[1]['transitTimeToNearestNode']
    featDict['transitTimeNearestNodeReverse'] = reverseHopDict['transitTimeToNearestNode']
    
    featDict['numBussesNearestNode1Hop'] = forwardHopDict[0]['numBussesToNearestNode']
    featDict['numBussesNearestNode2Hop'] = forwardHopDict[1]['numBussesToNearestNode']
    featDict['numBussesNearestNodeReverse'] = reverseHopDict['numBussesToNearestNode']
    t1 = time.time()
    timeChecks['Add Hop Features'] = t1 - t0
    
    #Interconnectivity
    
    t0 = time.time()
    
    oneHopForwardNodes = list(set(forwardHopDict[0]['transitNodes']))
    reverseHopNodes = list(set(reverseHopDict['reachableNodes']))
    
    countIntersection = 0
    intDistancetoD = []
    intDistancetoO = []
    
    #TODO - check this logic doesn't seem right
    if len(oneHopForwardNodes) > 0 and len(reverseHopNodes):
    
        oneHopForwardLatLon = itemgetter(*oneHopForwardNodes)(oaLatLons)
        reverseHopLatLon = itemgetter(*reverseHopNodes)(oaLatLons)
        
        if len(oneHopForwardNodes) == 1:
            oneHopForwardLatLon = [oneHopForwardLatLon]
            
        if len(reverseHopNodes) == 1:
            reverseHopLatLon = [reverseHopLatLon]
        
        timeTransitFromOrigin = []
        numBussesFromOrigin = []
        timeTransitToDest = []
        numBussesToDest = []
        
        forwardIndex = -1
        for f in oneHopForwardLatLon:
            if isinstance(f,float):
                f = oneHopForwardLatLon
            forwardIndex += 1
            reverseIndex = -1
            for r in reverseHopLatLon:
                if isinstance(r,float):
                    r = reverseHopLatLon
                reverseIndex += 1
                if haversine(f[0],f[1],r[0],r[1]) < threshold:
                    countIntersection += 1
                    try:
                        bussesFromOrigin = OAtoOABus[originOA][oneHopForwardNodes[forwardIndex]]
                    except:
                        busOriginFound = False
                        walkIndex = 0
                        while busOriginFound == False:
                            try:
                                bussesFromOrigin = OAtoOABus[walkableRadiusOrigin[walkIndex]][oneHopForwardNodes[forwardIndex]]
                                busOriginFound = True
                            except:
                                walkIndex += 1
                    #Get buses to destination - searc walkable OAs around dest if can't find direct bus
                    try:
                        bussesToDest = OAtoOABus[reverseHopNodes[reverseIndex]][destOA]
                    except:
                        busDestFound = False
                        walkIndex = 0
                        while busDestFound == False:
                            try:
                                bussesToDest = OAtoOABus[reverseHopNodes[reverseIndex]][walkableRadiusDest[walkIndex]]
                                busDestFound = True
                            except:
                                walkIndex += 1
                    intDistancetoD.append(haversine(f[1],f[0],destLon,destLat))
                    intDistancetoO.append(haversine(f[1],f[0],originLon,originLat))
                    
                    timeTransitFromOrigin.append(statistics.mean(bussesFromOrigin))
                    numBussesFromOrigin.append(len(bussesFromOrigin))
                    timeTransitToDest.append(statistics.mean(bussesToDest))
                    numBussesToDest.append(len(bussesFromOrigin))
    else:
        pass
    
    t1 = time.time()
    timeChecks['Calculating Intersections'] = t1 - t0
    
    t0 = time.time()
    featDict['numIntersections'] = countIntersection
    if len(intDistancetoD) > 0:
        featDict['minIntersectionDistD'] = min(intDistancetoD)
        featDict['meanIntersectionDistD'] = statistics.mean(intDistancetoD)
        featDict['transitTimeIntersectionToD'] = timeTransitToDest[np.argmin(intDistancetoD)]
        featDict['numBussesIntersectionToD'] = numBussesToDest[np.argmin(intDistancetoD)]
    else:
        featDict['minIntersectionDistD'] = None
        featDict['meanIntersectionDistD'] = None    
        featDict['transitTimeIntersectionToD'] = 0
        featDict['numBussesIntersectionToD'] = 0
    
    if len(intDistancetoO) > 0:
        featDict['minIntersectionDistO'] = min(intDistancetoO)
        featDict['meanIntersectionDistO'] = statistics.mean(intDistancetoO)
        featDict['transitTimeIntersectionToO'] = timeTransitFromOrigin[np.argmin(intDistancetoO)]
        featDict['numBussesIntersectionToO'] = numBussesFromOrigin[np.argmin(intDistancetoO)]
    else:
        featDict['minIntersectionDistO'] = None
        featDict['meanIntersectionDistO'] = None
        featDict['transitTimeIntersectionToO'] = 0
        featDict['numBussesIntersectionToO'] = 0
        
    t1 = time.time()
    timeChecks['Intersections Features'] = t1 - t0
    
    t0 = time.time()
    if len(OAinBbox) > 0:
        featDict['propForewardReachableBbox'] = len(set(forwardHopDict[0]['reachalbeNodesInBox'])) / len(OAinBbox)
        featDict['propForewardReachableBbox2'] = len(set(forwardHopDict[1]['reachalbeNodesInBox'])) / len(OAinBbox)
        featDict['propReverseReachableBbox'] = len(set(reverseHopDict['reachalbeNodesInBox'])) / len(OAinBbox)
    else:
        featDict['propForewardReachableBbox'] = 1
        featDict['propForewardReachableBbox2'] = 1
        featDict['propReverseReachableBbox'] = 1
    t1 = time.time()
    timeChecks['Reecord reachable features'] = t1 - t0
    
    return featDict,timeChecks

#%%

# def getOaTOOaFeatVec(originOA,destOA,oa_info,boxBuffer,OAtoOABus,wm_oas,oasReachableNMin,reverseOASearch,oaLats,oaLons,threshold):
    
#     featDict = {}
#     timeChecks = {}

#     t0 = time.time()
#     # Walking radius'
#     walkableRadiusOrigin = oasReachableNMin[originOA]
#     if originOA not in walkableRadiusOrigin:
#         walkableRadiusOrigin.append(originOA)
    
#     walkableRadiusDest = oasReachableNMin[destOA]
#     if destOA not in walkableRadiusDest:
#         walkableRadiusDest.append(destOA)
#     t1 = time.time()
#     timeChecks['Walking Radius'] = t1 - t0

#     #Get Initial Data    
#     #1 hops
#     t0 = time.time()
#     forwardHopNodes, transitNodes = forwardHop(originOA, OAtoOABus, walkableRadiusOrigin)
#     reverseHopNodes = reverseHop(destOA, reverseOASearch, walkableRadiusDest,oasReachableNMin)
#     t1 = time.time()
#     timeChecks['Cal 1 hop'] = t1 - t0
    
#     #2hops
#     t0 = time.time()
#     twoHopNodes, transitNodesTemp = forwardHop(originOA, OAtoOABus, walkableRadiusOrigin)
#     for o in transitNodes:
#         forwardHopNodesTemp, transitNodesTemp = forwardHop(o, OAtoOABus, walkableRadiusOrigin)
#         twoHopNodes = twoHopNodes + forwardHopNodesTemp
#     twoHopNodes = list(set(twoHopNodes))
#     t1 = time.time()
#     timeChecks['Cal 2 Hops'] = t1 - t0
    
    
#     t0 = time.time()
#     destLat = oa_info[oa_info['oa_id'] == destOA]['oa_lat'].values[0]
#     destLon = oa_info[oa_info['oa_id'] == destOA]['oa_lon'].values[0]
    
#     originLat = oa_info[oa_info['oa_id'] == originOA]['oa_lat'].values[0]
#     originLon = oa_info[oa_info['oa_id'] == originOA]['oa_lon'].values[0]
    
#     # #Get bounding box
#     # bbox = getBbox(originLat, destLat, originLon, destLon, boxBuffer, boxBuffer)
#     # t1 = time.time()
#     # timeChecks['Cal Bounding Box'] = t1 - t0

#     # t0 = time.time()
#     # OAinBbox = []
#     # for i, r in wm_oas.iterrows():
#     #     pass
#     #     if r['geometry'].intersects(bbox):
#     #         OAinBbox.append(r['OA11CD'])
#     # t1 = time.time()
#     # timeChecks['Cal Bounding Box Intersections'] = t1 - t0


#     OAinBbox = getBoundingBoxOAs(originLat,originLon,destLat, destLon,boxBuffer,oaLats,oaLons,oa_info)
#     t1 = time.time()
#     timeChecks['Bounding Box Interactions'] = t1 - t0

#     #Calculate Features
#     #1 hop reachability
#     t0 = time.time()
#     connected1Hop = False
#     for o in walkableRadiusOrigin:
#         if o in reverseHopNodes:
#             connected1Hop = True    
#     featDict['connect1Hop'] = int(connected1Hop)
#     t1 = time.time()
#     timeChecks['1 Hop Feat'] = t1 - t0
    
#     #2 hop connectivity
#     t0 = time.time()
#     connected2Hops = False
#     for o in twoHopNodes:
#         if o == destOA:
#             connected2Hops = True
#         elif o in walkableRadiusDest:
#             connected2Hops = True
#     featDict['connect2Hops'] = int(connected2Hops)
#     t1 = time.time()
#     timeChecks['2 Hop Feat'] = t1 - t0
    
#     # Distance after 1 hop
#     t0 = time.time()
#     forwardHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(forwardHopNodes)][['oa_lat','oa_lon']])
#     distances1Hop = []
#     for i in forwardHopLatLon:
#         distances1Hop.append(haversine(i[1], i[0], destLon, destLat))    
#     featDict['minDist1Hop'] = min(distances1Hop)
#     t1 = time.time()
#     timeChecks['Dist after 1 hop'] = t1 - t0
    
#     # Distance to destination after 2 hops
#     t0 = time.time()
#     twoHopHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(twoHopNodes)][['oa_lat','oa_lon']])    
#     distances2Hops = []
#     for i in twoHopHopLatLon:
#         distances2Hops.append(haversine(i[1], i[0], destLon, destLat))    
#     featDict['minDist2Hop'] = min(distances2Hops)
#     t1 = time.time()
#     timeChecks['Dist after 2 hops'] = t1 - t0
    
#     #Distance to origin after 1 reverse hop
#     t0 = time.time()
#     reverseHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(reverseHopNodes)][['oa_lat','oa_lon']])
#     distances1HopReverse = []
#     for i in reverseHopLatLon:
#         distances1HopReverse.append(haversine(i[1], i[0], originLon, originLat))    
#     featDict['distances1RevHop'] = min(distances1HopReverse)
#     t1 = time.time()
#     timeChecks['Cal Reverse Hop'] = t1 - t0

#     #interconnectivity
    
#     # #Create buffer around each point
#     # t0 = time.time()
#     # forwardHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(forwardHopNodes)][['oa_lat','oa_lon']])
    
#     # forwardBuffers = []
#     # for i in forwardHopLatLon:
#     #     forwardBuffers.append(Point(i[1],i[0]).buffer(0.001))
#     # t1 = time.time()
#     # timeChecks['Buffer Forward Hops'] = t1 - t0
    
#     # #Create buffer around each point
#     # t0 = time.time()
#     # reverseHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(reverseHopNodes)][['oa_lat','oa_lon']])
#     # reverseBuffers = []
#     # for i in reverseHopLatLon:
#     #     reverseBuffers.append(Point(i[1],i[0]).buffer(0.001))
#     # t1 = time.time()
#     # timeChecks['Buffer Reverser Hop'] = t1 - t0
   
#     # t0 = time.time()
#     # countIntersection = 0
#     # intDistancetoD = []
#     # intDistancetoO = []
#     # fInd = 0
    
#     # for f in forwardBuffers:
#     #     for r in reverseBuffers:
#     #         if f.intersects(r):
#     #             countIntersection += 1
#     #             fLat = oa_info[oa_info['oa_id'] == forwardHopNodes[fInd]]['oa_lat'].values[0]
#     #             fLon = oa_info[oa_info['oa_id'] == forwardHopNodes[fInd]]['oa_lon'].values[0]
#     #             intDistancetoD.append(haversine(fLon,fLat,destLon,destLat))
#     #             intDistancetoO.append(haversine(fLon,fLat,originLon,originLat))            
#     #     fInd += 1
#     # t1 = time.time()
#     # timeChecks['Calculating Intersections'] = t1 - t0
    
#     t0 = time.time()
#     forwardHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(forwardHopNodes)][['oa_lat','oa_lon']])
#     reverseHopLatLon = np.array(oa_info[oa_info['oa_id'].isin(reverseHopNodes)][['oa_lat','oa_lon']])
    
#     countIntersection = 0
#     intDistancetoD = []
#     intDistancetoO = []
    
#     for f in forwardHopLatLon:
#         for r in reverseHopLatLon:
#             if haversine(f[0],f[1],r[0],r[1]) < threshold:
#                 countIntersection += 1
#                 intDistancetoD.append(haversine(f[1],f[0],destLon,destLat))
#                 intDistancetoO.append(haversine(f[1],f[0],originLon,originLat))  
    
#     t1 = time.time()
#     timeChecks['Calculating Intersections'] = t1 - t0
    
#     t0 = time.time()
#     featDict['numIntersections'] = countIntersection
#     if len(intDistancetoD) > 0:
#         featDict['minIntersectionDistD'] = min(intDistancetoD)
#         featDict['meanIntersectionDistD'] = statistics.mean(intDistancetoD)
#     else:
#         featDict['minIntersectionDistD'] = None
#         featDict['meanIntersectionDistD'] = None    
#     if len(intDistancetoO) > 0:
#         featDict['minIntersectionDistO'] = min(intDistancetoO)
#         featDict['meanIntersectionDistO'] = statistics.mean(intDistancetoO)
#     else:
#         featDict['minIntersectionDistO'] = None
#         featDict['meanIntersectionDistO'] = None
#     t1 = time.time()
#     timeChecks['Intersections Features'] = t1 - t0


#     t0 = time.time()
#     #Percent reachable nodes (anywhere)
#     propForwardReachable1 = len(forwardHopNodes)/ oa_info.shape[0]    
#     propForwardReachable2 = len(twoHopNodes)/ oa_info.shape[0]    
#     propReverseReachable = len(reverseHopNodes)/ oa_info.shape[0]
#     featDict['propForwardReachable1'] = propForwardReachable1
#     featDict['propForwardReachable2'] = propForwardReachable2
#     featDict['propReverseReachable'] = propReverseReachable
#     t1 = time.time()
#     timeChecks['Reachability features'] = t1 - t0

#     #Percent reachable nodes in bounding box
#     t0 = time.time()
#     #1 forward hop
#     countForwardinBox1 = 0
#     for o in forwardHopNodes:
#         if o in OAinBbox:
#             countForwardinBox1 += 1
#     if len(OAinBbox) > 0:
#         propForewardReachableBbox = countForwardinBox1/ len(OAinBbox)
#     else:
#         propForewardReachableBbox = 1
#     t1 = time.time()
#     timeChecks['Calculate Percent Reachable 1 hop'] = t1 - t0
    
#     t0 = time.time()
#     #2 forward hop
#     countForwardinBox2 = 0
#     for o in twoHopNodes:
#         if o in OAinBbox:
#             countForwardinBox2 += 1
            
#     if len(OAinBbox) > 0:
#         propForewardReachableBbox2 = countForwardinBox2/ len(OAinBbox)
#     else:
#         propForewardReachableBbox2 = 1
#     t1 = time.time()
#     timeChecks['Calculate Percent Reachable 2 hops'] = t1 - t0
    
#     t0 = time.time()
#     #1 reverse hop
#     countReverseinBox = 0
    
#     for o in reverseHopNodes:
#         if o in OAinBbox:
#             countReverseinBox += 1
    
#     if len(OAinBbox) > 0:
#         propReverseReachableBbox = countReverseinBox/ len(OAinBbox)
#     else:
#         propReverseReachableBbox = 1
#     t1 = time.time()
#     timeChecks['Calculate Percent Reachable 1 reverse hop'] = t1 - t0

#     t0 = time.time()
#     featDict['propForewardReachableBbox'] = propForewardReachableBbox
#     featDict['propForewardReachableBbox2'] = propForewardReachableBbox2
#     featDict['propReverseReachableBbox'] = propReverseReachableBbox
#     t1 = time.time()
#     timeChecks['Reecord reachable features'] = t1 - t0
    
#     return featDict,timeChecks

#%%

def staticFeatures(OPPairs,oa_info,wm_oas,urbanCentre,OAsReachableOnFoot,featureVector):
    euclidDist = []
    sizeO = []
    sizeD = []
    
    popDensO = []
    popDensD = []
    
    distToUrbCentO = []
    distToUrbCentD = []
    
    walkin4 = []
    walkin8 = []
    walkin12 = []
    walkin16 = []
    walkin20 = []
    
    
    for i, r in OPPairs.iterrows():
        euclidDist.append(haversine(oa_info[oa_info['oa_id'] == r['oa_id']]['oa_lon'],oa_info[oa_info['oa_id'] == r['oa_id']]['oa_lat'],oa_info[oa_info['oa_id'] == r['poi_oa']]['oa_lon'],oa_info[oa_info['oa_id'] == r['poi_oa']]['oa_lat']))
        sizeO.append(wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0])
        sizeD.append(wm_oas[wm_oas['OA11CD'] == r['poi_oa']]['Shape__Are'].values[0])
        
        popDensO.append(oa_info[oa_info['oa_id'] == r['oa_id']]['age_all_residents'].values[0] / wm_oas[wm_oas['OA11CD'] == r['oa_id']]['Shape__Are'].values[0])
        popDensD.append(oa_info[oa_info['oa_id'] == r['poi_oa']]['age_all_residents'].values[0] / wm_oas[wm_oas['OA11CD'] == r['poi_oa']]['Shape__Are'].values[0])
        distToUrbCentO.append(haversine(oa_info[oa_info['oa_id'] == r['oa_id']]['oa_lon'],oa_info[oa_info['oa_id'] == r['oa_id']]['oa_lat'],urbanCentre[1],urbanCentre[0]))
        distToUrbCentD.append(haversine(oa_info[oa_info['oa_id'] == r['poi_oa']]['oa_lon'],oa_info[oa_info['oa_id'] == r['poi_oa']]['oa_lat'],urbanCentre[1],urbanCentre[0]))
        
        d = r['poi_oa']
        
        # under4 = 0
        # under8 = 0
        # under12 = 0
        # under16 = 0
        # under20 = 0
        
        for j in [4,8,12,16,20]:
            if d in OAsReachableOnFoot[r['oa_id']][j]:
                locals()['walkin'+str(j)].append(1)
            else:
                locals()['walkin'+str(j)].append(0)
        
        # walkin4.append(under4)
        # walkin8.append(under8)
        # walkin12.append(under12)
        # walkin16.append(under16)
        # walkin20.append(under20)
    
    featureVector['euclidDist'] = euclidDist
    featureVector['sizeO'] = sizeO
    featureVector['sizeD'] = sizeD
    featureVector['popDensO'] = popDensO
    featureVector['popDensD'] = popDensD
    featureVector['distToUrbCentO'] = distToUrbCentO
    featureVector['distToUrbCentD'] = distToUrbCentD
    featureVector['walkin4'] = walkin4
    featureVector['walkin8'] = walkin8
    featureVector['walkin12'] = walkin12
    featureVector['walkin16'] = walkin16
    featureVector['walkin20'] = walkin20
    
    return featureVector

#%%

def originCenteredFeatures(originsIndex,oaLatLons,OAsReachableOnFoot,walkableMins,OAtoOABus,wm_oas,oa_info,ODPairs,params,generateOBTree,featuresLoc):
    
    if generateOBTree:
        OBTree = {}
    else:
        f = open(featuresLoc+params['OBTreeFile'], 'rb')
        OBTree = pickle.load(f)
        f.close()
    hopFeats1 = []
    hopFeats2 = []
    
    count = 0
    for o in originsIndex:
        count += 1
        #print((count/len(originsIndex))*100)
        originLatLon = oaLatLons[o]
        
        if generateOBTree:
            OBTree[o] = {}
            #Create empty tree
            for h in range(2):
                OBTree[o][h+1] = {}
            
            #Construct the OB Tree
            
            #First layer hops
            #Get zones walkable
            walkableRadiusOrigin = OAsReachableOnFoot[o][walkableMins]
            #Record visited nodes
            visitedNodes = []
            #Iterate through walkable zones
            for a in walkableRadiusOrigin:
                #Add to tree, with 0 and 0 for transit metrics
                OBTree[o][1][a] = [o,0,0]
                visitedNodes.append(a)
                for k,v in OAtoOABus[a].items():
                    if k not in visitedNodes:
                        #Add to tree
                        OBTree[o][1][k] = [o,statistics.mean(v),len(v)]
                        visitedNodes.append(k)
            
            #Second layer hops
            for k,v in OBTree[o][1].items():
                #Get walking OA from each leaf
                walkableRadiusOrigin = OAsReachableOnFoot[k][walkableMins]
                #Visit each node
                for a2 in walkableRadiusOrigin:
                    #If not already visisted add to tree
                    if a2 not in visitedNodes:
                        OBTree[o][2][a2] = [k,v[1],v[2]]
                        visitedNodes.append(a2)
                    #Go to all transit nodes from here
                    for k2,v2 in OAtoOABus[a2].items():
                        if k2 not in visitedNodes:
                            OBTree[o][2][k2] = [a2,v[1]+statistics.mean(v2),len(v2)]
                            visitedNodes.append(a2)
    
        #Get static features
        OASize = wm_oas[wm_oas['OA11CD'] == o]['Shape__Are'].values[0]
        popDens = oa_info[oa_info['oa_id'] ==o]['age_all_residents'].values[0] / wm_oas[wm_oas['OA11CD'] == o]['Shape__Are'].values[0]
        distWayPoint = haversine(oa_info[oa_info['oa_id'] == o]['oa_lon'],oa_info[oa_info['oa_id'] == o]['oa_lat'],params['urbanCentre'][1],params['urbanCentre'][0])
    
        #Get all associated D
        destinations = list(ODPairs[ODPairs['oa_id'] == o]['poi_oa'].drop_duplicates())
        
        #Run KNN on D to OAs in OB Hop
        
        destLatLons = []
        for d in destinations:
            destLatLons.append(oaLatLons[d])
        destLatLons = np.array(destLatLons)
        
        #1 Hop
        
        OBLatLons = pd.DataFrame(oaLatLons).T.loc[list(OBTree[o][1].keys())].values
        
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(OBLatLons)
        nearestOBPoints = knn.kneighbors(destLatLons, return_distance=False)
        
        for i in range(len(destinations)):
            #Generate features to describe relatioship between 1hop connecetivity and desination    
            dOa = destinations[i]
            dLatLon = destLatLons[i]
            
            #Test is destination walkable within certain time spans
            # walkin4 = 0
            # walkin8 = 0
            # walkin12 = 0
            # walkin16 = 0
            # walkin20 = 0
            walkableIn = []
            for j in [4,8,12,16,20]:
                if dOa in OAsReachableOnFoot[o][j]:
                    walkableIn.append(1)
                else:
                    walkableIn.append(0)
            reachable1 = dOa in OBTree[o][1].keys()
            nearestOBPoint = OBLatLons[nearestOBPoints[i]].squeeze()
            distToDest1 = haversine(dLatLon[0],dLatLon[1],nearestOBPoint[0],nearestOBPoint[1])
            distToOrigin1 = haversine(originLatLon[0],originLatLon[1],nearestOBPoint[0],nearestOBPoint[1])
            journeyDetails1 = OBTree[o][1][list(OBTree[o][1].keys())[nearestOBPoints[i][0]]]
            odDist = haversine(dLatLon[0],dLatLon[1],originLatLon[0],originLatLon[1])
            hopFeats1.append({
                'o':o,
                'd':dOa,
                'numNodes1Hop':len(OBTree[o][1].keys()),
                'sizeO':OASize,
                'popDensO':popDens,
                'distWPO':distWayPoint,
                'odDist':odDist,
                'walkin4':walkableIn[0],
                'walkin8':walkableIn[1],
                'walkin12':walkableIn[2],
                'walkin16':walkableIn[3],
                'walkin20':walkableIn[4],
                'destReachable1Hop':reachable1,
                'distToDest1Hop':distToDest1,
                'distToOrigin1Hop':distToOrigin1,
                'transitTimeNearestNode1Hop':journeyDetails1[1],
                'numBussesNearestNode1Hop':journeyDetails1[2]
                })
        
        #2 Hops
        
        OBLatLons = pd.DataFrame(oaLatLons).T.loc[list(OBTree[o][2].keys())].values
        if len(OBLatLons) > 0:
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(OBLatLons)
            nearestOBPoints = knn.kneighbors(destLatLons, return_distance=False)
            
            for i in range(len(destinations)):
                #Generate features to describe relatioship between 2hop connecetivity and desination    
                dOa = destinations[i]
                dLatLon = destLatLons[i]    
                reachable = dOa in OBTree[o][2].keys()
                nearestOBPoint = OBLatLons[nearestOBPoints[i]].squeeze()
                distToDest = haversine(dLatLon[0],dLatLon[1],nearestOBPoint[0],nearestOBPoint[1])
                distToOrigin = haversine(originLatLon[0],originLatLon[1],nearestOBPoint[0],nearestOBPoint[1])
                journeyDetails = OBTree[o][2][list(OBTree[o][2].keys())[nearestOBPoints[i][0]]]
                transitTime = journeyDetails[1]
                numBuses = journeyDetails[2]
                hopFeats2.append({
                    'o':o,
                    'd':dOa,
                    'numNodes2Hop':len(OBTree[o][2].keys()) + len(OBTree[o][1].keys()),
                    'destReachable2Hop':dOa in OBTree[o][2].keys(),
                    'distToDest2Hop':distToDest,
                    'distToOrigin2Hop':distToOrigin,
                    'transitTimeNearestNode2Hop':journeyDetails[1],
                    'numBussesNearestNode2Hop':journeyDetails[2]
                    })
        else:
            for i in range(len(destinations)):
                dOa = destinations[i]
                hopFeats2.append({
                    'o':o,
                    'd':dOa,
                    'numNodes2Hop':len(OBTree[o][1].keys()),
                    'destReachable2Hop':reachable1,
                    'distToDest2Hop':distToDest1,
                    'distToOrigin2Hop':distToOrigin1,
                    'transitTimeNearestNode2Hop':journeyDetails1[1],
                    'numBussesNearestNode2Hop':journeyDetails1[2]
                    })
        
    return OBTree,hopFeats1,hopFeats2

#%%

def destinationCenteredFeatured(destinationIndexes,oaLatLons,OAsReachableOnFoot,walkableMins,reverseOASearch,OAtoOABus,wm_oas,oa_info,ODPairs,params):
    IBFeats = []
    
    IBTree = {}
    for d in destinationIndexes:
        
        destLatLon = oaLatLons[d]
        
        IBTree[d] = {}
        for h in range(1):
            IBTree[d][h+1] = {}
    
        walkableRadiusDest = OAsReachableOnFoot[d][walkableMins]
    
        for a in walkableRadiusDest:
            for t in reverseOASearch[a]:
                bussesToNode = OAtoOABus[t][a]
                IBTree[d][1][t] = [t,statistics.mean(bussesToNode),len(bussesToNode)]
    
    
        #Get static features
        OASize = wm_oas[wm_oas['OA11CD'] == d]['Shape__Are'].values[0]
        popDens = oa_info[oa_info['oa_id'] ==d]['age_all_residents'].values[0] / wm_oas[wm_oas['OA11CD'] == d]['Shape__Are'].values[0]
        distWayPoint = haversine(oa_info[oa_info['oa_id'] == d]['oa_lon'],oa_info[oa_info['oa_id'] == d]['oa_lat'],params['urbanCentre'][1],params['urbanCentre'][0])
    
        #Get all associated origins
        origins = list(ODPairs[ODPairs['poi_oa'] == d]['oa_id'].drop_duplicates())
    
        # Get Lat Lons
        orgLatLons = []
        for o in origins:
            orgLatLons.append(oaLatLons[o])
        orgLatLons = np.array(orgLatLons)
    
        #1 Hop - get lat lons
    
        IBLatLons = []
    
        for dest in IBTree[d][1].keys():
            IBLatLons.append(oaLatLons[dest])
        IBLatLons = np.array(IBLatLons)
    
        #Run KNN
        if len(IBLatLons)>0:
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(IBLatLons)
            nearestIBPoints = knn.kneighbors(orgLatLons, return_distance=False)
        
            for i in range(len(origins)):
                #An origin point assocaited to fixed POI
                oOa = origins[i]
                oLatLon = orgLatLons[i]
                #For origin point, closest 1hop OA to destination
                nearestIBPoint = IBLatLons[nearestIBPoints[i]].squeeze()
                distToDest = haversine(destLatLon[0],destLatLon[1],nearestIBPoint[0],nearestIBPoint[1])
                distToOrigin = haversine(oLatLon[0],oLatLon[1],nearestIBPoint[0],nearestIBPoint[1])
                journeyDetails = IBTree[d][1][list(IBTree[d][1].keys())[nearestIBPoints[i][0]]]
                transitTime = journeyDetails[1]
                numBuses = journeyDetails[2]
                IBFeats.append({
                    'o':oOa,
                    'd':d,
                    'numNodesReverse':len(IBTree[d][1].keys()),
                    'sizeD':OASize,
                    'popDensD':popDens,
                    'distWPD':distWayPoint,
                    'destReachableReverse':oOa in IBTree[d][1].keys(),
                    'distToDestReverse':distToDest,
                    'distToOriginReverse':distToOrigin,
                    'transitTimeNearestNodeReverse':transitTime,
                    'numBussesNearestNodeReverse':numBuses
                    })
        else:
            for i in range(len(origins)):
                oOa = origins[i]
                IBFeats.append({
                    'o':oOa,
                    'd':d,
                    'numNodesReverse':0,
                    'sizeD':OASize,
                    'popDensD':popDens,
                    'distWPD':distWayPoint,
                    'destReachableReverse':False,
                    'distToDestReverse':0,
                    'distToOriginReverse':0,
                    'transitTimeNearestNodeReverse':0,
                    'numBussesNearestNodeReverse':0
                    })
    return IBFeats,IBTree

#%%

def ODcentredFeatured(ODPairs,oaLatLons,OBTree,IBTree,boxBuffer,oa_info,OAsReachableOnFoot):
    
    count = 0 
    intersectionFeatures = []
    for OD in ODPairs[['oa_id','poi_oa']].drop_duplicates().values:
        count += 1
        #print((count/len(ODPairs))*100)
        o = OD[0]
        d = OD[1]
        oLatLon = np.array(oaLatLons[o])
        dLatLon = np.array(oaLatLons[d])
        
        #Get Bounding Box
        bBox = getBoundingBoxOAs(oLatLon[0],oLatLon[1],dLatLon[0], dLatLon[1],boxBuffer,oa_info)
        
        if len(bBox) > 0:
            #1hop OB in bbox
            nodesInBox1Hop = len(list(set(list(OBTree[o][1].keys())) & set(bBox))) / len(bBox)
            #2hop OB in bbox
            nodesInBox2Hop = len(list(set(list(OBTree[o][2].keys())) & set(bBox))) / len(bBox)
        else:
            nodesInBox1Hop = 0
            nodesInBox2Hop = 0
        
        #1hop Rev in bbox
        try:
            nodesInBox1HopRev = len(list(set(list(IBTree[d][1].keys())) & set(bBox))) / len(bBox)
        except:
            nodesInBox1HopRev = 0
        
        #Intersections
        intersectionLatLons,intersections,IBintersections,OBintersections,OBLatLons,IBLatLons,numImpactIntersectons = getIntersections(o,d,OBTree,IBTree,oaLatLons,bBox,OAsReachableOnFoot)
    
        #Percentage high impact intersections
        if len(intersections) > 0:
            highImpactIntersections = numImpactIntersectons / len(intersections)
        else:
            highImpactIntersections = 0
    
        if len(intersections) > 0:
    
            originMetrics = np.zeros((len(intersections),3))
            destMetrics = np.zeros((len(intersections),3))
            
            metricIndex = 0
            
            for i in intersections:
                #Get route o to intersection
                oToIntsct = OBTree[o][1][i[0]]
                #Get intersection to d routes
                dFromIntsct = IBTree[d][1][i[1]]
                #Get distance between o and intersection
                oToIDist = haversine(oaLatLons[o][0],oaLatLons[o][1],oaLatLons[i[0]][0],oaLatLons[i[0]][1])
                #Get distance between intersection and d
                iToDDist = haversine(oaLatLons[d][0],oaLatLons[d][1],oaLatLons[i[1]][0],oaLatLons[i[1]][1])
                
                originMetrics[metricIndex,:] = [oToIntsct[1],oToIntsct[2],oToIDist]
                destMetrics[metricIndex,:] = [dFromIntsct[1],dFromIntsct[2],iToDDist]
                metricIndex += 1
                
            intersectionFeatures.append({
                'o':o,
                'd':d,
                'nodesInBox1Hop':nodesInBox1Hop,
                'nodesInBox2Hop':nodesInBox2Hop,
                'nodesInBox1HopRev':nodesInBox1HopRev,
                'numIntersections':len(intersections),
                'highImpactIntersections':highImpactIntersections,
                'originTravelTimeMin':originMetrics[:,0].min(),
                'originTravelTimeMean':originMetrics[:,0].mean(),
                'originRouteFreqMax':originMetrics[:,1].max(),
                'originRouteFreqMean':originMetrics[:,1].mean(),
                'originDistMin':originMetrics[:,2].min(),
                'originDistMean':originMetrics[:,2].mean(),
                'destTravelTimeMin':destMetrics[:,0].min(),
                'destTravelTimeMean':destMetrics[:,0].mean(),
                'destRouteFreqMax':destMetrics[:,1].max(),
                'destRouteFreqMean':destMetrics[:,1].mean(),
                'destDistMin':destMetrics[:,2].min(),
                'destDistMean':destMetrics[:,2].mean(),
                'ttToClosestDIntersection':originMetrics[:,0][np.argmin(destMetrics[:,0])],
                'rfToClosestDIntersection':originMetrics[:,1][np.argmin(destMetrics[:,0])],
                'ttFromClosestOIntersection':destMetrics[:,0][np.argmin(originMetrics[:,0])],
                'rfFromClosestOIntersection':destMetrics[:,1][np.argmin(originMetrics[:,0])],
                'ttToMostFrequentDIntersection':originMetrics[:,0][np.argmax(destMetrics[:,1])],
                'rfToMostFrequentDIntersection':originMetrics[:,1][np.argmax(destMetrics[:,1])],
                'ttFromMostFrequentOIntersection':destMetrics[:,0][np.argmax(originMetrics[:,1])],
                'rfFromMostFrequentOIntersection':destMetrics[:,1][np.argmax(originMetrics[:,1])]
                })
    
        else:
            intersectionFeatures.append({
                'o':o,
                'd':d,
                'nodesInBox1Hop':nodesInBox1Hop,
                'nodesInBox2Hop':nodesInBox2Hop,
                'nodesInBox1HopRev':nodesInBox1HopRev,
                'numIntersections':0,
                'highImpactIntersections':highImpactIntersections,
                'originTravelTimeMin':0,
                'originTravelTimeMean':0,
                'originRouteFreqMax':0,
                'originRouteFreqMean':0,
                'originDistMin':0,
                'originDistMean':0,
                'destTravelTimeMin':0,
                'destTravelTimeMean':0,
                'destRouteFreqMax':0,
                'destRouteFreqMean':0,
                'destDistMin':0,
                'destDistMean':0,
                'ttToClosestDIntersection':0,
                'rfToClosestDIntersection':0,
                'ttFromClosestOIntersection':0,
                'rfFromClosestOIntersection':0,
                'ttToMostFrequentDIntersection':0,
                'rfToMostFrequentDIntersection':0,
                'ttFromMostFrequentOIntersection':0,
                'rfFromMostFrequentOIntersection':0
                })
    
    return intersectionFeatures
