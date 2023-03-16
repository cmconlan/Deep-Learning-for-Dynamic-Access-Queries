import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
import statistics
import random
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D

#Import Results
allResults = pd.read_csv('G:/My Drive/University/Working Folder/Transport Access Tool/SSR-Access-Query/results/results_exps_1_to_8.csv', index_col = 0)

resultsAM = allResults[allResults['Stratum'] == 'AM Peak']
resultsOLvl = resultsAM[resultsAM['level'] == 'O']
resultsOLvl = resultsOLvl[resultsOLvl['Demo Group'] == 'lone_parent_total']
resultsBirm = resultsOLvl[resultsOLvl['Area'] == 'Birm']
resultsCov = resultsOLvl[resultsOLvl['Area'] == 'Cov']

resultsBirmJT = resultsBirm[resultsBirm['cost'] == 'JT']
resultsBirmGAC = resultsBirm[resultsBirm['cost'] != 'JT']

resultsCovJT = resultsCov[resultsCov['cost'] == 'JT']
resultsCovGAC = resultsCov[resultsCov['cost'] != 'JT']

models = ['COREG','GNN','Mean Teacher','MLPRegression','OLSRegression']

labelDict = {
    'COREG' : 'COREG',
    'GNN' : 'GNN',
    'Mean Teacher' : 'MT',
    'MLPRegression' : 'MLP',
    'OLSRegression' : 'OLS'
    }

poiDict = {
    'School':'School',
    'Hospital':'Hospital',
    'Vaccination Centre':'Vaccine Center',
    'Job Centre':'Job Center'
    }

metricDict = {
    'Cost Correlation' : 'MAC Corr (%)',
    'Std Correlation':'ACSD Corr (%)',
    'Accuracy' : 'Accuracy (%)',
    'Jains Error' : 'FIE'
    }

POIs = ['School','Hospital','Vaccination Centre','Job Centre']

perfMetrics = ['Cost Correlation','Std Correlation','Accuracy','Jains Error']

outputFile = 'G:/My Drive/University/Working Folder/Transport Access Tool/Paper/Figures/'
resultsLoc = 'E:/Data/results/'

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

#%% Plot 1

#4 by 2
#Row 1 - birmingham, each plot shows JT mean error over different budgets for each POI
#Row 1 - coventry, each plot shows JT mean error over different budgets for each POI


fig, ax = plt.subplots(2,4, figsize=(14,5.5))

fontsizeSmall = 15
fontsizeLarge = 16

#Row 1
downCount = 0
acrossCount = 0
#Plot1

yLimUpper = 13
yLimLower = int((resultsBirmJT['Cost Error']/60).min())

for p in POIs:

    
    print(p)
    print(acrossCount)
    
    poiResults = resultsBirmJT[resultsBirmJT['POI'] == p]
    
    ax[downCount,acrossCount].set_ylim([yLimLower, yLimUpper])
    
    for m in models:
        test = poiResults[poiResults['model'] == m]
        test['Error (Minutes)'] = test['Cost Error']/60
        test['budget'] = test['budget']*100
        test.plot('budget','Error (Minutes)', ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')
        
    ax[downCount,acrossCount].set_xlabel('Budget (%)', fontsize = fontsizeSmall)
    #ax[downCount,acrossCount].set_ylabel('Journey Time \n Error (Minutes)')
    ax[downCount,acrossCount].set_title('Birmingham \n ' + poiDict[p], fontsize = fontsizeLarge)
    ax[downCount,acrossCount].tick_params(axis='both', which='major', labelsize=fontsizeSmall)
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

ax[downCount,0].set_ylabel('Journey Time \n Error (Minutes)', fontsize = fontsizeSmall)


#Row 2
downCount = 1
acrossCount = 0
#Plot1

yLimUpper = 13
yLimLower = int((resultsCovJT['Cost Error']/60).min())

for p in POIs:

    
    print(p)
    print(acrossCount)
    
    poiResults = resultsCovJT[resultsCovJT['POI'] == p]
    
    ax[downCount,acrossCount].set_ylim([yLimLower, yLimUpper])
    
    for m in models:
        test = poiResults[poiResults['model'] == m]
        test['Error (Minutes)'] = test['Cost Error']/60
        test['budget'] = test['budget']*100
        test.plot('budget','Error (Minutes)', ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')
        
    ax[downCount,acrossCount].set_xlabel('Budget (%)', fontsize = fontsizeSmall)
    #ax[downCount,acrossCount].set_ylabel('Journey Time \n Error (Minutes)')
    ax[downCount,acrossCount].set_title('Coventry \n ' + poiDict[p], fontsize = fontsizeLarge)
    ax[downCount,acrossCount].tick_params(axis='both', which='major', labelsize=fontsizeSmall)
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

ax[downCount,0].set_ylabel('Journey Time \n Error (Minutes)', fontsize = fontsizeSmall)

#plt.legend()

fig.legend(ax[downCount,3].get_legend_handles_labels()[0],ax[downCount,3].get_legend_handles_labels()[1],loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=5, fontsize = fontsizeSmall)

plt.tight_layout()
fig.savefig(outputFile+'jt_birm_cov.pdf')
plt.show()

#%%



#%%


fontsizeSmall = 15
fontsizeLarge = 16


fig, ax = plt.subplots(2,4, figsize=(14,5.5))

#Row 1
downCount = 0
acrossCount = 0
#Plot1

p = 'Vaccination Centre'

poiResults = resultsBirmGAC[resultsBirmGAC['POI'] == p]

poiResults['budget'] = poiResults['budget']*100

for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget (%)', fontsize = fontsizeSmall)    
    ax[downCount,acrossCount].set_title('Birmingham - ' + metricDict[metric], fontsize = fontsizeLarge)
    ax[downCount,acrossCount].tick_params(axis='both', which='major', labelsize=fontsizeSmall)
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)


#Row 2
downCount = 1
acrossCount = 0
#Plot1

poiResults = resultsCovGAC[resultsCovGAC['POI'] == p]
poiResults['budget'] = poiResults['budget']*100

for metric in perfMetrics:
    for m in models:
        print()
        print()
        print(metric)
        print(m)
        print(poiResults[poiResults['model'] == m][metric])
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget (%)', fontsize = fontsizeSmall)    
    ax[downCount,acrossCount].set_title('Coventry - ' + metricDict[metric], fontsize = fontsizeLarge)
    ax[downCount,acrossCount].tick_params(axis='both', which='major', labelsize=fontsizeSmall)
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

#plt.suptitle(poiDict[p],fontsize = fontsizeLarge)

fig.legend(ax[downCount,3].get_legend_handles_labels()[0],ax[downCount,3].get_legend_handles_labels()[1],loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=5, fontsize = fontsizeSmall)


plt.tight_layout()
fig.savefig(outputFile+'performance_vc.pdf')
plt.show()

#%%

fig, ax = plt.subplots(2,4, figsize=(12,5.5))

#Row 1
downCount = 0
acrossCount = 0
#Plot1

p = 'School'
poiResults = resultsBirmJT[resultsBirmGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

#Row 1
downCount = 1
acrossCount = 0
#Plot1

poiResults = resultsCovJT[resultsCovGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)
plt.suptitle(poiDict[p])
plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(2,4, figsize=(12,5.5))

#Row 1
downCount = 0
acrossCount = 0
#Plot1
p = 'Hospital'
poiResults = resultsBirmJT[resultsBirmGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

#Row 1
downCount = 1
acrossCount = 0
#Plot1

poiResults = resultsCovJT[resultsCovGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)

plt.suptitle(poiDict[p])
plt.tight_layout()
plt.show()

#%%

fig, ax = plt.subplots(2,4, figsize=(12,5.5))

#Row 1
downCount = 0
acrossCount = 0
#Plot1
p = 'Job Centre'
poiResults = resultsBirmJT[resultsBirmGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)
ax[downCount,3].legend().set_visible(False)

#Row 1
downCount = 1
acrossCount = 0
#Plot1

poiResults = resultsCovJT[resultsCovGAC['POI'] == p]


for metric in perfMetrics:
    for m in models:
        poiResults[poiResults['model'] == m].plot('budget',metric, ax = ax[downCount,acrossCount], label = labelDict[m], marker = '*')

    
    ax[downCount,acrossCount].set_xlabel('Budget')    
    ax[downCount,acrossCount].set_title(metricDict[metric])
    
    acrossCount += 1

ax[downCount,0].legend().set_visible(False)
ax[downCount,1].legend().set_visible(False)
ax[downCount,2].legend().set_visible(False)

plt.suptitle(poiDict[p])
plt.tight_layout()
plt.show()

#%%

#%%


#%% Plot accessibility for some examples

scheme='quantiles'
exp = 'exp3'
b = 0.03
level = 'O'
model = 'MLPRegression'
cost = 'GAC'
params = paramsDict[exp]

wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]

f = open(resultsLoc + exp +'.txt', 'rb')
resultsDict = pickle.load(f)
f.close()

indexO = resultsDict[exp]['O']['index']
indexO.sort()

expResults = resultsDict[exp][level][b][model]
expTrainMask = resultsDict[exp][level][b]['trainMask']
expTestMask = resultsDict[exp][level][b]['testMask']
labelCost = resultsDict[exp][level]['labellingCosts']
groundTruth = resultsDict[exp][level]['yAct']
scalerY = resultsDict[exp][level]['scalerY']

#%%

yPred = expResults['yPred']

predInd = 0
predicted = []

for i in range(len(expTestMask)):
    if expTestMask[i]:
        predicted.append(scalerY.inverse_transform(yPred[predInd].reshape(1, -1))[0])
        predInd += 1
    else:
        predicted.append(scalerY.inverse_transform(groundTruth[i].reshape(1, -1))[0])


predictedMean = np.array(predicted)[:,0]


modelResults = pd.DataFrame(index = indexO)
modelResults['Predicted'] = predictedMean
modelResults['Actual'] = scalerY.inverse_transform(groundTruth)[:,0]

wm_oas_birm = wm_oas.merge(modelResults,left_on = 'OA11CD',right_index = True,how = 'inner')

#%%

# fig, axs = plt.subplots(1,2,figsize=(10,5))

# wm_oas.plot(column='Predicted', cmap='OrRd', scheme=scheme, ax = axs[0])
# axs[0].set_title('Predicted')
# wm_oas.plot(column='Actual', cmap='OrRd', scheme=scheme, ax = axs[1])
# axs[1].set_title('Actual')

# plt.show()

#%%
#exp3, exp7

#%%

scheme='quantiles'
exp = 'exp7'
b = 0.1
level = 'O'
model = 'MLPRegression'
cost = 'GAC'
params = paramsDict[exp]

wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]

f = open(resultsLoc + exp +'.txt', 'rb')
resultsDict = pickle.load(f)
f.close()

indexO = resultsDict[exp]['O']['index']
indexO.sort()

expResults = resultsDict[exp][level][b][model]
expTrainMask = resultsDict[exp][level][b]['trainMask']
expTestMask = resultsDict[exp][level][b]['testMask']
labelCost = resultsDict[exp][level]['labellingCosts']
groundTruth = resultsDict[exp][level]['yAct']
scalerY = resultsDict[exp][level]['scalerY']

#%%

yPred = expResults['yPred']

predInd = 0
predicted = []

for i in range(len(expTestMask)):
    if expTestMask[i]:
        predicted.append(scalerY.inverse_transform(yPred[predInd].reshape(1, -1))[0])
        predInd += 1
    else:
        predicted.append(scalerY.inverse_transform(groundTruth[i].reshape(1, -1))[0])


predictedMean = np.array(predicted)[:,0]


modelResults = pd.DataFrame(index = indexO)
modelResults['Predicted'] = predictedMean
modelResults['Actual'] = scalerY.inverse_transform(groundTruth)[:,0]

wm_oas_cov = wm_oas.merge(modelResults,left_on = 'OA11CD',right_index = True,how = 'inner')

#%%

wm_oas_birm['Predicted'][wm_oas_birm['Predicted'] < 0] = 0
wm_oas_cov['Predicted'][wm_oas_cov['Predicted'] < 0] = 0


#%%

titleFontSize = 13

fig, axs = plt.subplots(2,2,figsize=(10,10))

wm_oas_birm.plot(column='Actual', cmap='OrRd', scheme=scheme, ax = axs[0,0], legend = True, legend_kwds={'loc' : 'upper left'})
axs[0,0].set_title('A. Birmingham - Ground Truth', fontsize = titleFontSize)

wm_oas_birm.plot(column='Predicted', cmap='OrRd', scheme=scheme, ax = axs[0,1], legend = True, legend_kwds={'loc' : 'upper left'})
axs[0,1].set_title('B. Birmingham - MLP Predicted', fontsize = titleFontSize)


wm_oas_cov.plot(column='Actual', cmap='OrRd', scheme=scheme, ax = axs[1,0], legend = True, legend_kwds={'loc' : 'lower right', 'bbox_to_anchor':(0.3, -0.1)})
axs[1,0].set_title('C. Coventry - Ground Truth', fontsize = titleFontSize)

wm_oas_cov.plot(column='Predicted', cmap='OrRd', scheme=scheme, ax = axs[1,1], legend = True, legend_kwds={'loc' : 'lower right', 'bbox_to_anchor':(0.3, -0.1)})
axs[1,1].set_title('D. Coventry - MLP Predicted', fontsize = titleFontSize)

#fig.colorbar(plt1)

axs[0,0].axis('off')
axs[0,1].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')

plt.tight_layout()
fig.savefig(outputFile+'vc_access_mapped.pdf')
plt.show()

#%%

fig.legend(ax[downCount,3].get_legend_handles_labels()[0],ax[downCount,3].get_legend_handles_labels()[1],loc='upper center', bbox_to_anchor=(0.5, 0.02), fancybox=True, shadow=True, ncol=5, fontsize = fontsizeSmall)


#%%

axs[0,1].get_legend_handles_labels()[1]

#%% Print Transit Trees and Intersections

scheme='quantiles'
exp = 'exp3'
params = paramsDict[exp]

featuresLoc = 'E:/Data/features/'

wm_oas = gpd.read_file(shpFileLoc)
wm_oas = wm_oas[wm_oas['LAD11CD'] == params['area']]
oa_info = pd.read_csv(oaInfoLoc)
oa_info = oa_info.merge(wm_oas[['OA11CD']], left_on = 'oa_id', right_on = 'OA11CD', how = 'inner')
#oaLatLons - dictionary associating oa to lat/lons
oaLatLons=dict([(i,[a,b]) for i, a,b in zip(oa_info['oa_id'],oa_info['oa_lat'],oa_info['oa_lon'])])

walkableMins = 6

#%% Possible OD pairings

# E00046149
# E00045915

# E00046805
# E00045139

# E00045140
# E00046622

# E00046961
# E00046840

# E00175772
# E00047904

#%%

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

#%%

match = True

while match == True:
    o = oa_info.iloc[random.randint(0, len(oa_info))]['oa_id']
    d = oa_info.iloc[random.randint(0, len(oa_info))]['oa_id']
    if o == d:
        match = True
    else:
        match = False

print(o)
print(d)

IBTree = {}

destLatLon = oaLatLons[d]

IBTree[d] = {}
for h in range(1):
    IBTree[d][h+1] = {}

walkableRadiusDest = OAsReachableOnFoot[d][walkableMins]

for a in walkableRadiusDest:
    for t in reverseOASearch[a]:
        bussesToNode = OAtoOABus[t][a]
        IBTree[d][1][t] = [t,statistics.mean(bussesToNode),len(bussesToNode)]

ob = OBTree[o][1]
ib = IBTree[d][1]

# Get Intersections

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

print()
print(OBintersections)

#%%

legend_elements = [
    Line2D([0], [0], marker='*', color='white', label='Origin', markerfacecolor='green', markersize=12),
    Line2D([0], [0], marker='*', color='white', label='Destination', markerfacecolor='blue', markersize=12),
    Line2D([0], [0], marker = ".", color='white', label='Outbound Transit Hop', markerfacecolor='yellowgreen', markersize=12),
    Line2D([0], [0], marker = ".", color='white', label='Inbound Transit Hop', markerfacecolor='dodgerblue', markersize=12),
    Line2D([0], [0], marker = "P", color='white', label='Interchange', markerfacecolor='hotpink', markersize=12)
    ]

oLatLon = oaLatLons[o]


allLats = []

for i in OBLatLons:
    allLats.append(i[0])

for i in IBLatLons:
    allLats.append(i[0])

allLons = []

for i in OBLatLons:
    allLons.append(i[1])

for i in IBLatLons:
    allLons.append(i[1])

upperLat = max(allLats) + (max(allLats)*0.0001)
lowerLat = min(allLats) - abs(min(allLats)*0.0001)

upperLon = max(allLons) + (abs(max(allLons))*0.01)
lowerLon = min(allLons) - (abs(min(allLons)) * 0.01)

#Identify any intersection

fig, axs = plt.subplots(1,figsize=(20,5))


wm_oas.plot(ax = axs, edgecolor = 'lightgrey',linewidth=0.1, color = 'white')

for i in OBLatLons:
    axs.scatter(i[1], i[0], color = 'yellowgreen', s = 10)
    
for i in IBLatLons:
    axs.scatter(i[1], i[0], color = 'dodgerblue', s = 10)

for i in intersectionLatLons:
    axs.scatter(i[1], i[0], color = 'hotpink', s = 10, marker = "+")

axs.set_xlim([lowerLon,upperLon])
axs.set_ylim([lowerLat,upperLat])

axs.scatter(oLatLon[1], oLatLon[0], color = 'green', s = 40, marker="*")
axs.scatter(destLatLon[1], destLatLon[0], color = 'blue', s = 40, marker= "*")

#axs.legend(handles=legend_elements,fontsize = '8', bbox_to_anchor=(0.75, 0.25))

axs.legend(handles=legend_elements,fontsize = '8')

plt.tight_layout()
fig.savefig(outputFile+'transit-tree-example_7.pdf')
plt.show()

#%%    
print(o)
print(d)
#%%




#%%

#%%

#%%

#%%

#%%

#%%

#%%