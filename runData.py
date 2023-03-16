from pipeline.dataPreparation import getTrainingData, loadAdj
from pipeline.modelEvaluation import evaluateModel
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

kDict = {
    'School':3,
    'Hospital':2,
    'Job Centre':2}

urbCentreDict = {
    'E08000025':(52.47806970328915, -1.8989517979775046),
    'E08000026':(52.4078358796818, -1.5137840014354358)
    }

# Environment variables
shpFileLoc = 'Data/west_midlands_OAs/west_midlands_OAs.shp'
oaInfoLoc = 'Data/oa_info.csv'
dbLoc= 'Data/access.db'

#Static Variables
#Demographic Grouping
d = 'lone_parent_total'
#Allowable minutes to walk to find a bus stop
walkableMins = 6
#Buffer around origin and destination points around which to draw a box (e.g., high impact box)
boxBuffer = 0.38
#Allowable distance between stops to class as intersection
threshold = 0.15
#size of validation set
vb = 0.3

#%% Output all data

expNum = 1
experimentParameters = {}

#Inputs

#Target variable for model (use "Access" or "Time")
for targetVar in ['Access','Time']:
    #POI for model
    for p in ['School','Hospital','Job Centre']:
        # kDict
        k = kDict[p]
        #Time stratum for model
        for stratum in ['Weekday (AM peak)','Saturday']:
            #Area
            for area in ['E08000025']:
                #Urban centre
                urbanCentre = urbCentreDict[area]
                #level ('OA' or 'OAPOI')
                for level in ['OAPOI']:
                    #budget
                    for b in [0.3,0.1,0.03]:
                        #Sample rate for each oa-poi relationship
                        for sr in [1,0.66,0.33]:
                            
                            print('Experiment : ' + str(expNum))
                            
                            experimentParameters[expNum] = {
                                'targetVar':targetVar,
                                'p':p,
                                'k':k,
                                'stratum':stratum,
                                'area':area,
                                'level':level,
                                'b':b,
                                'sr':sr}

                            #Get Training Data
                            x, y, yAct, scalerX, scalerY, testMask, valMask, trainMask, OPTrips, OPPairs, featureVector, oa_info, wm_oas, oaMask, labeledMask, unlabeledMask = getTrainingData(p, stratum, targetVar, area, walkableMins, boxBuffer, threshold, sr, b, vb, shpFileLoc, oaInfoLoc, dbLoc, urbanCentre, level,k)
                            
                            outputDir = 'Data/test_data/'+str(expNum)
                            
                            if not os.path.exists(outputDir):
                                os.makedirs(outputDir)
                            
                            np.save(outputDir+'/x.npy', x)
                            np.save(outputDir+'/y.npy', y)
                            np.save(outputDir+'/yAct.npy', yAct)
                            np.save(outputDir+'/trainMask.npy', trainMask)
                            np.save(outputDir+'/testMask.npy', testMask)
                            np.save(outputDir+'/valMask.npy', valMask)
                            np.save(outputDir+'/labeledMask.npy', labeledMask)
                            np.save(outputDir+'/unlabeledMask.npy', unlabeledMask)
                            oa_info['oa_id'].to_csv(outputDir + '/oa_index.csv')
                            OPPairs.to_csv(outputDir + '/OPPairs.csv')
                            
                            f = open(outputDir+'/scalerX.txt', 'wb')
                            pickle.dump(scalerX,f)
                            f.close()
                            
                            f = open(outputDir+'/scalerY.txt', 'wb')
                            pickle.dump(scalerY,f)
                            f.close()
                            
                            f = open('expParams.txt', 'wb')
                            pickle.dump(experimentParameters,f)
                            f.close()
                            
                            expNum += 1

#%% Run Models and Evaluate



#%% OLS

allResults = []

from methods.OLS import OLSRegression

predVector, infTime = OLSRegression(x,y,trainMask,testMask)
modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=True)

print(modelResults)

#%% MLP
from methods.MLPRegression import MLPRegression

hiddenMLP1 = 2000
hiddenMLP2 = 1000
hiddenMLP3 = 500
hiddenMLP4 = 250
epochsMLP = 250
dp = 0.05
device = 'cpu'

#MLP
predVector, infTime, losses, lossesVal = MLPRegression(x,y,trainMask,testMask,valMask,1000, 500, 250, 100,epochsMLP, device,dp)

print(losses[-1])
print(min(lossesVal))

plt.plot(losses)
plt.plot(lossesVal)
plt.show()
#Evaluate MLP
modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=True)

print(modelResults)

modelResultsName = {'method':'MLP'}
modelResultsName.update(modelResults)

allResults.append(modelResultsName)

#%% GNN at OA Level

level = 'OA'
#Get Training Data
x, y, yAct, scalerX, scalerY, testMask, valMask, trainMask, OPTrips, OPPairs, featureVector, oa_info, wm_oas, oaMask = getTrainingData(p, stratum, targetVar, area, walkableMins, boxBuffer, threshold, sr, b, vb, shpFileLoc, oaInfoLoc, dbLoc, urbanCentre, level)
#Load Adjacency Matrix
edgeIndexNp,edgeWeightsNp = loadAdj(oaMask, area)

from methods.GNN import runGNN

#GNN Paramaeters
device = 'cpu'
hidden1 = 32
hidden2 = 32
k = 1
dp = 0.33
epochs = 100

predVector, infTime, losses, lossesVal, lossesTest = runGNN(x,y,device,edgeIndexNp,edgeWeightsNp,hidden1,hidden2,epochs,trainMask,testMask,valMask,k, dp)

print(losses[-1])
print(min(lossesVal))
print(min(lossesTest))

plt.plot(losses)
plt.plot(lossesVal)
plt.plot(lossesTest)
plt.show()

#%% GNN At OAPOI Level



#%% COREG

from methods.COREG import CoregTrainer

num_train = 100
num_trials = 1

coregTrainer = CoregTrainer(num_train,num_trials,x,y,trainMask,testMask)
coregTrainer.run_trials()
predVector = coregTrainer.test_hat

modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=True)
print(modelResults)

#%%



#%% SSDKL





#%%




#%%



#%%




#%%



#%%