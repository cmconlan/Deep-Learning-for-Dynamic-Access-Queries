import statsmodels.api as sm
import numpy as np
import time

def OLSRegression(x,y,trainMask,testMask):
    
    timeStart = time.time()
    model = sm.OLS(y[trainMask], x[trainMask]).fit()
    
    predictedAccessCost = []
    for i in x[testMask]:
        predictedAccessCost.append(model.predict(i)[0])
    timeEnd = time.time()
    return np.array(predictedAccessCost), timeEnd-timeStart