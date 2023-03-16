import torch
from torch_geometric.nn import ChebConv, GCNConv, SGConv
from torch.nn import Linear
import torch.nn.functional as F
import time
import numpy as np
from copy import deepcopy
import gc

class GCN(torch.nn.Module):
    def __init__(self, numFeatures, hidden_channel1,hidden_channel2,lin_channel1, lin_channel2, outputDim, k, dp):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # Initialize the layers
        #GN Conv
        # self.conv1 = GCNConv(numFeatures, hidden_channel1, improved = True)
        # self.conv2 = GCNConv(hidden_channel1, hidden_channel2, improved = True)
        #Cheb Conv
        self.conv1 = ChebConv(numFeatures, hidden_channel1, K = k)
        self.conv2 = ChebConv(hidden_channel1, hidden_channel2, K = k)
        # self.conv3 = ChebConv(hidden_channel2, hidden_channel3, K = k)
        #SGConv
        # self.conv1 = SGConv(numFeatures, hidden_channel1)
        # self.conv2 = SGConv(hidden_channel1, hidden_channel2)
        #Cluster Conv
        #self.conv1 = ClusterGCNConv(numFeatures, hidden_channel1)
        #self.conv2 = ClusterGCNConv(hidden_channel1, hidden_channel2)
        
        #Linea Layers
        self.lin1 = torch.nn.Linear(hidden_channel2, lin_channel1)
        self.lin2 = torch.nn.Linear(lin_channel1, lin_channel2)
        
        self.out = Linear(lin_channel2, outputDim)

    #def forward(self, x, edge_index):
    def forward(self, x, edge_index, edgeWeights, dp):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index, edgeWeights)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index, edgeWeights)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)
        
        # Third Message Passing Layer
        # x = self.conv3(x, edge_index, edgeWeights)
        # x = x.relu()
        # x = F.dropout(x, p= dp, training=self.training)
        
        # # Linear Channel 1
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)

        # Linear Channel 1
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p= dp, training=self.training)
        
        # Output layer 
        x = self.out(x)
        x = x.relu()
        return x
    
#%%

def runGNN(x,y,device,edgeIndexNp,edgeWeightsNp,hidden1,hidden2,linear1,linear2,epochs,trainMask,testMask,valMask,k, dp):
    t0 = time.time()
    #Attach data to tensor
    _x = torch.tensor(x).to(device).float()
    _y = torch.tensor(y).to(device).float()
    
    edgeIndex = torch.tensor(edgeIndexNp).to(device).long()
    edgeWeights = torch.tensor(np.expand_dims(edgeWeightsNp,1)).to(device).float()
    
    #Instantitate moedl
    model = GCN(numFeatures = _x.shape[1], hidden_channel1=hidden1, hidden_channel2=hidden2, lin_channel1=linear1, lin_channel2=linear2,outputDim = 1, k=k, dp=dp)
    model = model.to(device)
    
    #Model settings
    # decay = 5e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=decay)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    criterion = torch.nn.MSELoss()

    model.train()

    losses = []
    lossesVal = []
    lossesTest = []

    modelsPreserve = {}

    for i in range(epochs):
        
        optimizer.zero_grad()
        #Predict
        yPred = model(_x,edgeIndex,edgeWeights,dp)
        
        #Calculate Loss
        loss = criterion(yPred[trainMask], _y[trainMask].unsqueeze(1))
        lossVal = criterion(yPred[valMask], _y[valMask].unsqueeze(1))
        lossTest = criterion(yPred[testMask], _y[testMask].unsqueeze(1))
        losses.append(loss.cpu().detach())
        lossesVal.append(lossVal.cpu().detach())
        lossesTest.append(lossTest.cpu().detach())
        #Backpropogate
        loss.backward()
        optimizer.step()
        #Preserve model weights
        modelsPreserve[i] = deepcopy(model).state_dict()
    
    bestValidationModel = np.argmin(lossesVal)
    model.load_state_dict(modelsPreserve[bestValidationModel])
        
    t1 = time.time()
    yPred = np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy())
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return yPred, t1-t0, losses, lossesVal,lossesTest

#%%

# import pandas as pd

# device = 'cpu'
# k = 1
# dp = 0.2
# epochs = 100
# hidden1exps = [1500,3000,6000]
# hidden2exps = [1000,2000,4000]
# linearexps = [500,1000,2000]
# epochs = [100,500,2000]

# #%%

# device = 'cpu'
# k = 1
# dp = 0.2
# epochs = 100
# hidden1exps = [1500,3000,6000]
# hidden2exps = [1000,2000,4000]
# linearexps = [500,1000,2000]
# epochs = [100,300]

# allResults = []

# count = -1

# for hidden1 in hidden1exps:
#     for hidden2 in hidden2exps:
#         for linear1 in linearexps:
#             for epoch in epochs:
                
#                 count+= 1
                
#                 print('Experiment : ' + str(count))
                
#                 t0 = time.time()
                
#                 _x = torch.tensor(x).to(device).float()
#                 _y = torch.tensor(y).to(device).float()

#                 edgeIndex = torch.tensor(edgeIndexNp).to(device).long()
#                 edgeWeights = torch.tensor(np.expand_dims(edgeWeightsNp,1)).to(device).float()

#                 #Instantitate moedl
#                 model = GCN(numFeatures = _x.shape[1], hidden_channel1=hidden1, hidden_channel2=hidden2, lin_channel1=linear1, outputDim = 1, k=k, dp=dp)
#                 model = model.to(device)

#                 #Model settings
#                 decay = 5e-4
#                 #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=decay)
#                 optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
#                 criterion = torch.nn.MSELoss()

#                 model.train()

#                 losses = []
#                 lossesVal = []
#                 lossesTest = []

#                 modelsPreserve = {}

#                 for i in range(epoch):
                    
#                     if i % 20 == 0:
#                         print('Epoch : ' + str(i))
                    
#                     optimizer.zero_grad()
#                     #Predict
#                     yPred = model(_x,edgeIndex,edgeWeights,dp)
                    
#                     #Calculate Loss
#                     loss = criterion(yPred[trainMask], _y[trainMask].unsqueeze(1))
#                     lossVal = criterion(yPred[valMask], _y[valMask].unsqueeze(1))
#                     lossTest = criterion(yPred[testMask], _y[testMask].unsqueeze(1))
#                     losses.append(loss.cpu().detach())
#                     lossesVal.append(lossVal.cpu().detach())
#                     lossesTest.append(lossTest.cpu().detach())
#                     #Backpropogate
#                     loss.backward()
#                     optimizer.step()
#                     #Preserve model weights
#                     modelsPreserve[i] = deepcopy(model).state_dict()
                
                
#                 t1 = time.time()
                
#                 print(losses[-1])
#                 print(min(lossesVal))
#                 print(min(lossesTest))

#                 plt.plot(losses,label = 'Train')
#                 plt.plot(lossesVal,label = 'Val')
#                 plt.plot(lossesTest,label = 'Test')
#                 plt.legend()
#                 plt.title('Hidden 1 : ' + str(hidden1) + ' Hidden 2 : ' + str(hidden2) + ' Linear : ' + str(linear1) + ' Epochs : ' + str(epoch))
#                 plt.show()
                
#                 #Get Last Model
#                 predVector = np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy())
#                 modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=False)
                
#                 modelResultsName = {'method':'GNN','model':'last' ,'hidden1': hidden1, 'hidden2': hidden2, 'linear': linear1,'time': t1-t0}

#                 modelResultsName.update(modelResults)

#                 allResults.append(modelResultsName)
                
#                 #Get best model (according to validation)
                
#                 bestValidationModel = np.argmin(lossesVal)
#                 model.load_state_dict(modelsPreserve[bestValidationModel])

#                 predVector = np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy())
#                 modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=False)
                
#                 modelResultsName = {'method':'GNN','model':'best' ,'hidden1': hidden1, 'hidden2': hidden2, 'linear': linear1,'time': t1-t0}

#                 modelResultsName.update(modelResults)

#                 allResults.append(modelResultsName)

# allResultspd = pd.DataFrame(allResults)

# #%%
# #GNN Paramaeters
# device = 'cpu'
# hidden1 = 4000
# hidden2 = 2000
# linear1 = 1000
# k = 1
# dp = 0.2
# epochs = 200
# d = 'disability_moderate'

# _x = torch.tensor(x).to(device).float()
# _y = torch.tensor(y).to(device).float()

# edgeIndex = torch.tensor(edgeIndexNp).to(device).long()
# edgeWeights = torch.tensor(np.expand_dims(edgeWeightsNp,1)).to(device).float()

# #Instantitate moedl
# model = GCN(numFeatures = _x.shape[1], hidden_channel1=hidden1, hidden_channel2=hidden2, lin_channel1=linear1, outputDim = 1, k=k, dp=dp)
# model = model.to(device)

# #Model settings
# decay = 5e-4
# #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=decay)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
# criterion = torch.nn.MSELoss()

# model.train()

# losses = []
# lossesVal = []
# lossesTest = []

# modelsPreserve = {}

# for i in range(epochs):
    
#     if i % 10 == 0:
#         print('Epoch : ' + str(i))
    
#     optimizer.zero_grad()
#     #Predict
#     yPred = model(_x,edgeIndex,edgeWeights,dp)
    
#     #Calculate Loss
#     loss = criterion(yPred[trainMask], _y[trainMask].unsqueeze(1))
#     lossVal = criterion(yPred[valMask], _y[valMask].unsqueeze(1))
#     lossTest = criterion(yPred[testMask], _y[testMask].unsqueeze(1))
#     losses.append(loss.cpu().detach())
#     lossesVal.append(lossVal.cpu().detach())
#     lossesTest.append(lossTest.cpu().detach())
#     #Backpropogate
#     loss.backward()
#     optimizer.step()
#     #Preserve model weights
#     modelsPreserve[i] = deepcopy(model).state_dict()

# print(losses[-1])
# print(min(lossesVal))
# print(min(lossesTest))

# plt.plot(losses,label = 'Train')
# plt.plot(lossesVal,label = 'Val')
# plt.plot(lossesTest,label = 'Test')
# plt.legend()
# plt.show()

# #%% Best Training Model

# bestTrainingModel = np.argmin(losses)
# model.load_state_dict(modelsPreserve[bestTrainingModel])

# predVector = np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy())
# modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=True)
# print(modelResults)

# #%%Best Validation Model

# bestValidationModel = np.argmin(lossesVal)
# model.load_state_dict(modelsPreserve[bestValidationModel])

# predVector = np.squeeze(model(_x, edgeIndex, edgeWeights, dp)[testMask].cpu().detach().numpy())
# modelResults = evaluateModel(testMask,scalerY,predVector,y,yAct,OPPairs,oa_info,d,wm_oas,mapResults=True)
# print(modelResults)

# #%%

# checkModelParams = list(model.parameters())

# #%%


# for epoch in range(0, epochs):
#     print(epoch)
#     optimizer.zero_grad()
#     #Predict
#     yPred = model(_x,edgeIndex,edgeWeights,dp)
#     #Calculate Loss
#     loss = criterion(yPred[trainMask], _y[trainMask].unsqueeze(1))
#     lossVal = criterion(yPred[valMask], _y[valMask].unsqueeze(1))
#     lossTest = criterion(yPred[testMask], _y[testMask].unsqueeze(1))
#     losses.append(loss.cpu().detach())
#     lossesVal.append(lossVal.cpu().detach())
#     lossesTest.append(lossTest.cpu().detach())
#     #Backpropogate
#     loss.backward()
#     optimizer.step()






























# #%%




# #%%




# #%%





# #%%