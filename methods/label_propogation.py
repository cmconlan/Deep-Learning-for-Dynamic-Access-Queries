import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist

#%%

def label_propagation_regression(X_l, y_l, X_u, X_val, y_val, sigma_2,outputPreds=False):
    # concatenate all the X's and y's
    #print(sigma_2)
    #print()
        
    X_all = np.concatenate([X_u,X_l,X_val], axis=0)
        
    #print("KNN init")
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(X_l, y_l)

    # y_u init
    # y_u = np.zeros((X_u.shape[0], ))
    y_u = knn.predict(X_u)

    # y_val_pred init
    # y_val_pred = np.zeros((X_val.shape[0], ))
    y_val_pred = knn.predict(X_val)
    
    y_all = np.concatenate([y_u,y_l,y_val_pred])

    # compute the kernel
    #print("Compute kernel")
    T = np.exp(-cdist(X_all, X_all, 'sqeuclidean') / sigma_2)
    # row normalize the kernel
    T /= np.sum(T, axis=1)[:, np.newaxis]

    #print("kernel done")
    delta = np.inf
    tol = 5e-6
    i = 0
    while delta > tol:
        y_all_new = T.dot(y_all)
        # clamp the labels known
        y_all_new[:X_l.shape[0]] = y_l
        delta = np.mean(y_all_new - y_all)
        y_all = y_all_new
        i += 1
        val_loss = np.sqrt(np.mean(np.square(y_all[-X_val.shape[0]:] - y_val)))
        if i % 10 == 0:
            pass
            #print("Iter {}: delta={}, val_loss={}".format(i, delta, val_loss))
        if i > 500:
            break

    # return final val loss
    if outputPreds:
        return val_loss, y_all
    else:
        return val_loss

def runLabelProp(x,y,testMask,valMask,trainMask,labeledMask,unlabeledMask):
        
    # X_labeled = x[labeledMask]
    # y_labeled = y[labeledMask]
    
    X_unlabeled = x[unlabeledMask]
    # y_unlabeled = y[unlabeledMask]
    
    X_train = x[trainMask]
    y_train = y[trainMask]
    
    X_test = x[testMask]
    y_test = y[testMask]
    
    X_val = x[valMask]
    y_val = y[valMask]
        
    fold_results = pd.DataFrame()
    fold = 0
    
    # search over sigma_2
    sigma_2s = np.linspace(0.8, 3.0, 5)

    val_losses = Parallel(n_jobs=1)(delayed(
        label_propagation_regression)(X_train, y_train, X_test, X_val, y_val, sigma_2)
        for sigma_2 in sigma_2s)
    
    best_idx = np.argmin(val_losses)
    
    best_val_loss = val_losses[best_idx]
    best_sigma_2 = sigma_2s[best_idx]
    fold_result_row = {
                       'fold': fold, 'best_val_loss': best_val_loss,
                       'best_sigma_2': best_sigma_2,
                       'sigma_2s': sigma_2s,
                       'val_losses': val_losses}
    fold_results = fold_results.append(fold_result_row, ignore_index=True)
    
    # test with the best
    val_loss, y_all = label_propagation_regression(X_train, y_train, X_test, X_val, y_val, best_sigma_2, True)
        
    return y_all[:testMask.sum()]