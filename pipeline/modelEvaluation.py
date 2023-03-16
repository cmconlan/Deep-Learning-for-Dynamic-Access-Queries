import numpy as np
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

#%%

def evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,target,mapResults=False,scheme='quantiles'):

    modelResults = {}    
    
    predInd = 0
    predicted = []
    
    for i in range(len(testMask)):
        if testMask[i]:
            predicted.append(scalerY.inverse_transform(predVector[predInd].reshape(1, -1))[0][0])
            predInd += 1
        else:
            predicted.append(scalerY.inverse_transform(y[i].reshape(1, -1))[0][0])
    
    predicted = np.array(predicted)
    
    # OA-POI Level
    
    error = abs(yAct.squeeze() - predicted)
    absError = error.mean()
    modelResults['OAPOI - Access Abs Error'] = absError
    
    # Get percentage error
    errorPct = error / yAct.squeeze()
    absErrorPcnt = errorPct.mean()
    modelResults['OAPOI - Access Abs Error Pcnt'] = absErrorPcnt
    
    #Get correlation coefficient
    modelResults['OAPOI - Access Correlation'] = pearsonr(predicted,yAct)[0][0]
    
    #OA Level
    
    #Aggregate at OA level
    temp = OPPairs.copy()
    temp['predicted'] = predicted
    oaLevelResults = temp.groupby('oa_id').mean()
    
    temp = OPPairs.copy()
    temp['actual'] = yAct
    oaLevelResults['actual'] = temp.groupby('oa_id').mean()['actual']
    
    oaLevelResults['error'] = abs(oaLevelResults['predicted'] - oaLevelResults['actual'])
    oaLevelResults['PcntError'] = oaLevelResults['error'] / oaLevelResults['actual']
    
    modelResults['OA - Access Abs Error'] = oaLevelResults['error'].mean()
    modelResults['OA - Access Abs Error Pcnt'] = oaLevelResults['PcntError'].mean()
    modelResults['OA - Access Correlation'] = pearsonr(oaLevelResults['predicted'].values,oaLevelResults['actual'].values)[0]
    
    #Jains error
    jainActual = (oaLevelResults['actual'].sum() ** 2) / ((oaLevelResults['actual']*oaLevelResults['actual']).sum() * oaLevelResults['actual'].shape[0])
    jainPred = (oaLevelResults['predicted'].sum() ** 2) / ((oaLevelResults['predicted']*oaLevelResults['predicted']).sum() * oaLevelResults['predicted'].shape[0])
    jainsError = jainActual - jainPred
    modelResults['OA - Jains Error'] = jainsError
    
    priorityIndex = oaLevelResults.copy()    
    population = oa_info[d].sum(axis=1)
    population.index = list(oa_info['oa_id'])
    population.name = 'population'
    priorityIndex = priorityIndex.merge(population,left_index = True, right_index = True, how = 'left')
    
    priorityIndex = priorityIndex[priorityIndex['population'] > 0]
    priorityIndex['PRM'] = priorityIndex['population'] / priorityIndex['population'].mean()
    priorityIndex['ARM_pred'] = priorityIndex['predicted'] / priorityIndex['predicted'].mean()
    priorityIndex['ARM_act'] = priorityIndex['actual'] / priorityIndex['actual'].mean()
    priorityIndex['nPRM'] = (priorityIndex['PRM'] - priorityIndex['PRM'].min()) / (
            priorityIndex['PRM'].max() - priorityIndex['PRM'].min())
    priorityIndex['nARM_pred'] = (priorityIndex['ARM_pred'] - priorityIndex['ARM_pred'].min()) / (
            priorityIndex['ARM_pred'].max() - priorityIndex['ARM_pred'].min())
    priorityIndex['nARM_act'] = (priorityIndex['ARM_act'] - priorityIndex['ARM_act'].min()) / (
            priorityIndex['ARM_act'].max() - priorityIndex['ARM_act'].min())
    priorityIndex['PI_pred'] = priorityIndex['nPRM'] * priorityIndex['nARM_pred']
    priorityIndex['PI_act'] = priorityIndex['nPRM'] * priorityIndex['nARM_act']
    
    priorityIndex['Rank_pred'] = priorityIndex['PI_pred'].rank(method='dense', ascending=False)
    priorityIndex['Rank_act'] = priorityIndex['PI_act'].rank(method='dense', ascending=False)
    
    priorityIndex['PI_act_corrected']=priorityIndex['Rank_act'].replace(0,priorityIndex['Rank_act'].values[np.nonzero(priorityIndex['Rank_act'].values)].min())
    
    
    #Calculate ARM, PRM etc to get at risk score
    #%Get Priority MAE
    priorityError = abs(priorityIndex['PI_act'] - priorityIndex['PI_pred']).values
    priorityMAE = priorityError.mean()
    modelResults['Priority Index -  MAE'] = priorityMAE
    
    #Get Priority MAPE
    # Get percentage error
    errorPct = priorityError / priorityIndex['PI_act_corrected'].values.squeeze()
    absErrorPcnt = errorPct.mean()
    modelResults['Priority Index -  MAPE'] = absErrorPcnt
    
    #Get correlation coefficient
    modelResults['Priority Index Correlation'] = pearsonr(priorityIndex['PI_pred'],priorityIndex['PI_act'])[0]
    
    #Map Accessibility
    
    #Append accesspred,acessact,prioritypred,priorityact to wmOAs
    if mapResults:
        wm_oas = wm_oas.merge(priorityIndex[['predicted','actual','PI_pred','PI_act']],left_on = 'OA11CD',right_index = True,how = 'inner')
        
        fig, axs = plt.subplots(2,2,figsize=(10,12))
        
        wm_oas.plot(column='predicted', cmap='OrRd', scheme=scheme, ax = axs[0,0])
        axs[0,0].set_title('Predicted : ' + str(target))
        wm_oas.plot(column='actual', cmap='OrRd', scheme=scheme, ax = axs[0,1])
        axs[0,1].set_title('Actual : ' + str(target))
        
        wm_oas.plot(column='PI_pred', cmap='OrRd', scheme=scheme, ax = axs[1,0])
        axs[1,0].set_title('Predicted Priotiy Areas')
        wm_oas.plot(column='PI_act', cmap='OrRd', scheme=scheme, ax = axs[1,1])
        axs[1,1].set_title('Actual Priotiy Areas')
    
        plt.show()
        
    return modelResults

#%%
# #%% OA Level

# modelResults = {}  

# predInd = 0
# predicted = []

# for i in range(len(testMask)):
#     if testMask[i]:
#         predicted.append(scalerY.inverse_transform(predVector[predInd].reshape(1, -1))[0][0])
#         predInd += 1
#     else:
#         predicted.append(scalerY.inverse_transform(y[i].reshape(1, -1))[0][0])

# predicted = np.array(predicted)

# #%%
# oaLevelResults = pd.DataFrame(index = list(oa_info['oa_id'][oaMask]))
# oaLevelResults['actual'] = list(yAct.squeeze())
# oaLevelResults['predicted'] = list(predicted)

# #%%

# oaLevelResults['error'] = abs(oaLevelResults['predicted'] - oaLevelResults['actual'])
# oaLevelResults['PcntError'] = oaLevelResults['error'] / oaLevelResults['actual']

# modelResults['OA - Access Abs Error'] = oaLevelResults['error'].mean()
# modelResults['OA - Access Abs Error Pcnt'] = oaLevelResults['PcntError'].mean()
# modelResults['OA - Access Correlation'] = pearsonr(oaLevelResults['predicted'].values,oaLevelResults['actual'].values)[0]

# #Jains error
# jainActual = (oaLevelResults['actual'].sum() ** 2) / ((oaLevelResults['actual']*oaLevelResults['actual']).sum() * oaLevelResults['actual'].shape[0])
# jainPred = (oaLevelResults['predicted'].sum() ** 2) / ((oaLevelResults['predicted']*oaLevelResults['predicted']).sum() * oaLevelResults['predicted'].shape[0])
# jainsError = jainActual - jainPred
# modelResults['OA - Jains Error'] = jainsError

# #%%

# priorityIndex = oaLevelResults.copy()
# priorityIndex = priorityIndex.merge(oa_info[['oa_id',d]].set_index('oa_id'),left_index = True, right_index = True, how = 'left')

# #%%
# priorityIndex = priorityIndex[priorityIndex[d] > 0]
# priorityIndex['PRM'] = priorityIndex[d] / priorityIndex[d].mean()
# priorityIndex['ARM_pred'] = priorityIndex['predicted'] / priorityIndex['predicted'].mean()
# priorityIndex['ARM_act'] = priorityIndex['actual'] / priorityIndex['actual'].mean()
# priorityIndex['nPRM'] = (priorityIndex['PRM'] - priorityIndex['PRM'].min()) / (
#         priorityIndex['PRM'].max() - priorityIndex['PRM'].min())
# priorityIndex['nARM_pred'] = (priorityIndex['ARM_pred'] - priorityIndex['ARM_pred'].min()) / (
#         priorityIndex['ARM_pred'].max() - priorityIndex['ARM_pred'].min())
# priorityIndex['nARM_act'] = (priorityIndex['ARM_act'] - priorityIndex['ARM_act'].min()) / (
#         priorityIndex['ARM_act'].max() - priorityIndex['ARM_act'].min())
# priorityIndex['PI_pred'] = priorityIndex['nPRM'] * priorityIndex['nARM_pred']
# priorityIndex['PI_act'] = priorityIndex['nPRM'] * priorityIndex['nARM_act']

# priorityIndex['Rank_pred'] = priorityIndex['PI_pred'].rank(method='dense', ascending=False)
# priorityIndex['Rank_act'] = priorityIndex['PI_act'].rank(method='dense', ascending=False)

# priorityIndex['PI_act_corrected']=priorityIndex['Rank_act'].replace(0,priorityIndex['Rank_act'].values[np.nonzero(priorityIndex['Rank_act'].values)].min())

# #%%

# #Calculate ARM, PRM etc to get at risk score
# #%Get Priority MAE
# priorityError = abs(priorityIndex['PI_act'] - priorityIndex['PI_pred']).values
# priorityMAE = priorityError.mean()
# modelResults['Priority Index -  MAE'] = priorityMAE

# #Get Priority MAPE
# # Get percentage error
# errorPct = priorityError / priorityIndex['PI_act_corrected'].values.squeeze()
# absErrorPcnt = errorPct.mean()
# modelResults['Priority Index -  MAPE'] = absErrorPcnt

# #Get correlation coefficient
# modelResults['Priority Index Correlation'] = pearsonr(priorityIndex['PI_pred'],priorityIndex['PI_act'])[0]


#%%


# modelResults = {}    

# predInd = 0
# predicted = []


# #Iterate through each instance
# for i in range(len(testMask)):
#     #When instance is a test instance
#     if testMask[i]:
#         #Take it from prediction vector
#         predicted.append(scalerY.inverse_transform(predVector[predInd].reshape(1, -1))[0][0])
#         predInd += 1
#     else:
#         #Else take the actual
#         predicted.append(scalerY.inverse_transform(y[i].reshape(1, -1))[0][0])

# predicted = np.array(predicted)


# #%%
# # OA-POI Level

# error = abs(yAct.squeeze() - predicted)
# absError = error.mean()
# modelResults['OAPOI - Access Abs Error'] = absError

# # Get percentage error
# errorPct = error / yAct.squeeze()
# absErrorPcnt = errorPct.mean()
# modelResults['OAPOI - Access Abs Error Pcnt'] = absErrorPcnt

# #Get correlation coefficient
# modelResults['OAPOI - Access Correlation'] = pearsonr(predicted,yAct)[0][0]

# #%%
# #OA Level

# #Aggregate at OA level
# temp = OPPairs.copy()
# temp['predicted'] = predicted
# oaLevelResults = temp.groupby('oa_id').mean()

# #%%
# temp = OPPairs.copy()
# temp['actual'] = yAct
# oaLevelResults['actual'] = temp.groupby('oa_id').mean()

# #%%
# oaLevelResults['error'] = abs(oaLevelResults['predicted'] - oaLevelResults['actual'])
# oaLevelResults['PcntError'] = oaLevelResults['error'] / oaLevelResults['actual']

# modelResults['OA - Access Abs Error'] = oaLevelResults['error'].mean()
# modelResults['OA - Access Abs Error Pcnt'] = oaLevelResults['PcntError'].mean()
# modelResults['OA - Access Correlation'] = pearsonr(oaLevelResults['predicted'].values,oaLevelResults['actual'].values)[0]

# #Jains error
# jainActual = (oaLevelResults['actual'].sum() ** 2) / ((oaLevelResults['actual']*oaLevelResults['actual']).sum() * oaLevelResults['actual'].shape[0])
# jainPred = (oaLevelResults['predicted'].sum() ** 2) / ((oaLevelResults['predicted']*oaLevelResults['predicted']).sum() * oaLevelResults['predicted'].shape[0])
# jainsError = jainActual - jainPred
# modelResults['OA - Jains Error'] = jainsError

# #%%
# priorityIndex = oaLevelResults.copy()

# #%%
# #Attach Demographic Population
# priorityIndex = priorityIndex.merge(oa_info[['oa_id',d]].set_index('oa_id'),left_index = True, right_index = True, how = 'left')

# #%%
# #Only select OAs with at least person in demographic group
# priorityIndex = priorityIndex[priorityIndex[d] > 0]
# #%%
# #Calculate Population Relative to Mean
# priorityIndex['PRM'] = priorityIndex[d] / priorityIndex[d].mean()
# #%%
# #Calculate Predict Accessibility relative to mean
# priorityIndex['ARM_pred'] = priorityIndex['predicted'] / priorityIndex['predicted'].mean()

# #%%
# #Calculate Actual Accessibility relative to mean
# priorityIndex['ARM_act'] = priorityIndex['actual'] / priorityIndex['actual'].mean()

# #%%
# #Min Max Normalise PRM
# priorityIndex['nPRM'] = (priorityIndex['PRM'] - priorityIndex['PRM'].min()) / (priorityIndex['PRM'].max() - priorityIndex['PRM'].min())

# #%%
# #Min Max Normalise ARM Pred
# priorityIndex['nARM_pred'] = (priorityIndex['ARM_pred'] - priorityIndex['ARM_pred'].min()) / (priorityIndex['ARM_pred'].max() - priorityIndex['ARM_pred'].min())

# #%%
# #Min Max Normalise ARM Actual
# priorityIndex['nARM_act'] = (priorityIndex['ARM_act'] - priorityIndex['ARM_act'].min()) / (priorityIndex['ARM_act'].max() - priorityIndex['ARM_act'].min())

# #%%

# pearsonr(priorityIndex['nARM_pred'].values,priorityIndex['nARM_act'].values)[0]

# #%%
# hist = priorityIndex['nPRM'].hist(bins=20)
# #%%
# hist = priorityIndex['nARM_pred'].hist(bins=20)
# #%%
# hist = priorityIndex['nARM_act'].hist(bins=20)

# #%%
# priorityIndex['PI_pred'] = priorityIndex['nPRM'] * priorityIndex['nARM_pred']
# #%%
# priorityIndex['PI_act'] = priorityIndex['nPRM'] * priorityIndex['nARM_act']

# #%%

# hist = priorityIndex['PI_pred'].hist(bins=20,alpha = 0.7)
# hist = priorityIndex['PI_act'].hist(bins=20,alpha = 0.7)

# #%%
# priorityIndex['Rank_pred'] = priorityIndex['PI_pred'].rank(method='dense', ascending=False)
# #%%
# priorityIndex['Rank_act'] = priorityIndex['PI_act'].rank(method='dense', ascending=False)
# #%%
# priorityIndex['PI_act_corrected']=priorityIndex['Rank_act'].replace(0,priorityIndex['Rank_act'].values[np.nonzero(priorityIndex['Rank_act'].values)].min())
# #%%

# #Calculate ARM, PRM etc to get at risk score
# #%Get Priority MAE
# priorityError = abs(priorityIndex['PI_act'] - priorityIndex['PI_pred']).values
# priorityMAE = priorityError.mean()
# modelResults['Priority Index -  MAE'] = priorityMAE

# #Get Priority MAPE
# # Get percentage error
# errorPct = priorityError / priorityIndex['PI_act_corrected'].values.squeeze()
# absErrorPcnt = errorPct.mean()
# modelResults['Priority Index -  MAPE'] = absErrorPcnt

# #Get correlation coefficient
# modelResults['Priority Index Correlation'] = pearsonr(priorityIndex['PI_pred'],priorityIndex['PI_act'])[0]


# #%%

# pearsonr(priorityIndex['Rank_pred'],priorityIndex['Rank_act'])[0]














































