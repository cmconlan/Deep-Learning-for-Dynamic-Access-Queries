import time
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import gc

# class Feedforward(torch.nn.Module):
#         def __init__(self, input_size, hidden_size):
#             super(Feedforward, self).__init__()
            
#             self.input_size = input_size
#             self.hidden_size  = hidden_size
            
#             self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#             self.fc2 = torch.nn.Linear(self.hidden_size, 1)
      
#         def forward(self, x):
#             hiddenLayer = self.fc1(x)
#             hiddenActi = hiddenLayer.relu()
            
#             hiddenActi = F.dropout(hiddenActi, p=0.1, training=self.training)
            
#             outputLayer = self.fc2(hiddenActi)
#             outputAct = outputLayer.relu()
#             return outputAct

#%%

#Model 1- MLP
#source - https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, dp):
            super(Feedforward, self).__init__()
            
            self.input_size = input_size
            self.hidden_size1  = hidden_size1
            self.hidden_size2  = hidden_size2
            self.hidden_size3  = hidden_size3
            self.hidden_size4  = hidden_size4
            
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size1)
            self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
            self.fc3 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)
            self.fc4 = torch.nn.Linear(self.hidden_size3, self.hidden_size4)
            self.fc5 = torch.nn.Linear(self.hidden_size4, 1)
      
        def forward(self, x, dp):
            hiddenLayer = self.fc1(x)
            dropOut1 = F.dropout(hiddenLayer, p=dp, training=self.training)
            
            hiddenLayer2 = self.fc2(dropOut1)
            dropOut2 = F.dropout(hiddenLayer2, p=dp, training=self.training)

            hiddenLayer3 = self.fc3(dropOut2)
            dropOut3 = F.dropout(hiddenLayer3, p=dp, training=self.training)
            
            hiddenLayer4 = self.fc4(dropOut3)
            dropOut4 = F.dropout(hiddenLayer4, p=dp, training=self.training)
            
            output = self.fc5(dropOut4)

            return output

#%%

def MLPRegression(x,y,trainMask,testMask,valMask,numHiddenLayers1,numHiddenLayers2,numHiddenLayers3,numHiddenLayers4,epochs, device, dp):
    timeStart = time.time()
    xTrain = torch.tensor(x[trainMask]).to(device).float()
    xTest = torch.tensor(x[testMask]).to(device).float()
    xVal = torch.tensor(x[valMask]).to(device).float()
    yTrain = torch.tensor(y[trainMask]).to(device).float()
    yVal = torch.tensor(y[valMask]).to(device).float()
    
    model = Feedforward(x.shape[1],numHiddenLayers1,numHiddenLayers2,numHiddenLayers3,numHiddenLayers4,dp)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    losses = []
    lossesVal = []
    
    model.train()
    
    modelsPreserve = {}
    
    for epoch in range(epochs):    
        
        optimizer.zero_grad()    # Forward pass
        y_pred = model(xTrain,dp)    # Compute Loss
        y_pred_val = model(xVal,dp)    # Compute Loss
        
        loss = criterion(y_pred.squeeze(), yTrain)
        lossVal = criterion(y_pred_val.squeeze(), yVal)
        
        losses.append(loss.cpu().detach())
        lossesVal.append(lossVal.cpu().detach())
        
        loss.backward()
        optimizer.step()
        modelsPreserve[epoch] = deepcopy(model).state_dict()

        # testTrain = float(losses[-1].cpu().detach().numpy()) - float(losses[0].cpu().detach().numpy())
        # if round(testTrain,3) != 0:
        #     proceed = True
    bestValidationModel = np.argmin(lossesVal)
    model.load_state_dict(modelsPreserve[bestValidationModel])
    timeEnd = time.time()
    yPred = np.squeeze(model(xTest,dp).cpu().detach().numpy())
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return yPred, timeEnd-timeStart, losses, lossesVal

#%%

# hiddenMLP1 = 2000
# hiddenMLP2 = 1000
# hiddenMLP3 = 500
# hiddenMLP4 = 250
# dp = 0.05
# device = 'cpu'

# xTrain = torch.tensor(x[trainMask]).to(device).float()
# xTest = torch.tensor(x[testMask]).to(device).float()
# xVal = torch.tensor(x[valMask]).to(device).float()

# yTrain = torch.tensor(y[trainMask]).to(device).float()
# yVal = torch.tensor(y[valMask]).to(device).float()
# yTest = torch.tensor(y[testMask]).to(device).float()

# #%%
# model = Feedforward(x.shape[1],hiddenMLP1,hiddenMLP2,hiddenMLP3,hiddenMLP4,dp)
# model.to(device)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# losses = []
# lossesVal = []

# model.train()

# #%%

# epoch = 0

# #%%
# optimizer.zero_grad()    # Forward pass
# yPred = model(xTrain,dp)    # Compute Loss
# yPredVal = model(xVal,dp)    # Compute Loss
# yPredTest = model(xTest,dp)

# #%% Histograms

# _ = plt.hist(yPred.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# _ = plt.hist(yTrain.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# plt.show()


# _ = plt.hist(yPredVal.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# _ = plt.hist(yVal.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# plt.show()

# _ = plt.hist(yPredTest.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# _ = plt.hist(yTest.cpu().detach().numpy(), bins='auto',alpha = 0.7)
# plt.show()

# #%% summary stats

# print('MEANS')
# print(yPred.cpu().detach().numpy().mean())
# print(yPredVal.cpu().detach().numpy().mean())
# print(yPredTest.cpu().detach().numpy().mean())
# print('STD')
# print(yPred.cpu().detach().numpy().std())
# print(yPredVal.cpu().detach().numpy().std())
# print(yPredTest.cpu().detach().numpy().std())

# #%%
# loss = criterion(yPred.squeeze(), yTrain)
# lossVal = criterion(yPredVal.squeeze(), yVal)
# lossTest = criterion(yPredTest.squeeze(), yTest)

# print(loss)
# print(lossVal)
# print(lossTest)

# #%%
# losses.append(loss.cpu().detach())
# lossesVal.append(lossVal.cpu().detach())

# #%%
# loss.backward()
# optimizer.step()