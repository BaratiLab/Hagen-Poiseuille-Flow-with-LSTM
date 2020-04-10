#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Nina Prakash
# CMU Mechanical and AI Lab
# Professor Farimani
# June 2019

# Use LSTM to predict 1D non-steady Hagen-Poiseuille flow.
# Assume constant pressure gradient and fluid starts at rest.


# In[2]:


import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


# In[3]:


# global variables

# date = '6-27-19'

numPoints = 16 # need this to pre-process
dt = .25 # s
ti = 0 # s
tf = 5 # s    # change back to 50 later
# rho = 888      # of oil at 20 degrees C [kg/m^3]
# mu = 0.8       # of oil at 20 degrees C [kg/m*s]
# nu = mu/rho
# dpdx = -10    # axial pressure gradient [Pa/m = kg/m^2*s^2]
# testSize = 0.2 # set aside 20% for testing

# Ds = np.linspace(0.5,1,num=10000) # increase later (500)


# ## Load Data

# In[4]:


data_dir = 'data/'
# numBCs = 10000
# minD = 0.50
# maxD = 1.00
# numTimeSteps = 20

# data_session_name = '%s_%.2f to %.2f_%s time steps' %(numBCs,minD,maxD,numTimeSteps)
data_session_name = 'all_params_50625'

dataset_filename = os.path.join(data_dir,'dataset_'+data_session_name+'.npy')
bc_filename = os.path.join(data_dir,'bc_'+data_session_name+'.npy')

target = np.load(dataset_filename)
inputs = np.load(bc_filename)

print('dataset loaded of size: ', np.shape(target))


# ## Pre-Process Data

# In[5]:


# replace first time step with all the boundary conditions?
# for now, hardcoded to fill in since there are 4 bc's and 17 points

for i in range(len(inputs)):
    D = inputs[i][0]
    dpdx = inputs[i][1]
    mu = inputs[i][2]
    nu = inputs[i][3]
    for j in range(numPoints+1):
        if 0 <= j < 5:
            target[i][0][j] = D
        elif 5 <= j < 9:
            target[i][0][j] = dpdx
        elif 9 <= j < 13:
            target[i][0][j] = mu
        else:
            target[i][0][j] = nu
        
print('replaced first time step with BCs\n')
# print('shape of inputs: ', np.shape(inputs))
# print('shape of target: ', np.shape(target))
# print('first bc: ', inputs[0])
# print('first target: ', target[0][0])
# print('another bc : ', inputs[7])
# print('another target: ', target[7][0])


# ## LSTM

# In[6]:


# adapted from Wei's code, model.py,
# and https://www.jessicayung.com/lstms-for-time-series-in-pytorch/

import torch
import torch.nn as nn
from torch.autograd import Variable

class BasicLSTM(nn.Module):
    
    def __init__(self,hidden_size, num_layers, num_features, device, dropout):
        super(BasicLSTM,self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_features = num_features
        self.device = device
        
        if self.num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        
        # define the LSTM layer
        self.lstm = nn.LSTM(
            input_size = self.num_features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True,
            dropout=self.dropout
        )
        
        # define the output layer
        self.dense = nn.Linear(self.hidden_size,self.num_features)
        
    # initialize hidden state as
    def initial_hidden_state(self,batch):
        return Variable(torch.zeros(self.num_layers,batch,self.hidden_size).to(self.device))

    # forward pass through LSTM layer
    def forward(self,x):
        batch, _, _ = x.shape
        h_0 = self.initial_hidden_state(batch)
        h_1 = self.initial_hidden_state(batch)
        out, _ = self.lstm(x, (h_0, h_1))
        out = self.dense(out)
        return out


# In[7]:


# from Wei's code, utils.py

def expanding_pred(net, truth_seq):
    pred_seq = np.empty(truth_seq.shape)
    pred_seq[0] = truth_seq[0]
    for i in range(1, len(pred_seq)):
        in_ = pred_seq[:i].reshape(1, i, -1)
        if type(in_) is not torch.Tensor:
            in_ = torch.Tensor(in_)
        if in_.dtype is not torch.float32:
            in_ = in_.type(torch.float32)
        out_ = net.predict_proba(in_)
        pred_seq[i] = out_.reshape(i, -1)[-1]
    return pred_seq


# In[ ]:


# adapted from Wei's code, main.py

import os
import time
import json
from torch.utils.data import DataLoader, random_split
# import torch.utils.data
# import torch
# print(torch.__version__)
import skorch
import pickle
from sklearn.model_selection import KFold

def main():
    start_time = time.time()
    torch.manual_seed(19)
    torch.cuda.manual_seed(19)
    np.random.seed(19)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)
    criterion = nn.MSELoss()
    
    data_dir = 'data/'
    session_name = '1000_epochs_adam_withcrossval'
    result_dir = 'result/' + session_name + '/'
    model_dir = 'model/' + session_name + '/'
    for dir_name in [result_dir, model_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
    num_points = 16
    num_features = num_points+1
    num_epochs = 1000 #500
    hidden_size = 196 #32
    num_layers=3
    dropout=0.2
    
    for j in [10000,20000,30000,40000,50000]:
        numBCName = str(j)
    
        print('NOW TRAINING WITH DATASET OF LENGTH ' + numBCName + '\n')
        # data_dir_ = os.path.join(data_dir, numBCName)
        resultDir = os.path.join(result_dir, numBCName)
        modelDir = os.path.join(model_dir,numBCName)
        for dir in  [resultDir, modelDir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
                
        dataset,foo = random_split(target, [j,len(target)-j])
        dataset = np.asarray(dataset)

        # above here is fine. for this size dataset, i need to train a model using k-fold cross val
        ###############
        
        # split dataset

#         test_len = int(0.2 * len(dataset))
#         valid_len = int(0.2 * (len(dataset) - test_len))
#         train_set, valid_set, test_set = random_split(dataset, [len(dataset) - valid_len - test_len, valid_len, test_len])

        kf = KFold(5)
        
        split_num = 0
        
        for train_i, test_i in kf.split(dataset):
            split_num += 1
            train = dataset[train_i]
            train_len = int(0.8*len(train))
            valid_len = int(0.2*len(train))
            train_set, valid_set = random_split(train, [train_len, valid_len])
            test_set = dataset[test_i]
            test_len = len(test_set)
            
            print('Running cross val split number ', split_num)
            print('shape of training set: ', np.shape(train_set))
            print('shape of validation set: ', np.shape(valid_set))
            print('shape of test set: ', np.shape(test_set))
            
            result_dir_ = os.path.join(resultDir, "Split %s" %(split_num))
            model_dir_ = os.path.join(modelDir, "Split %s" %(split_num))
            for dir in  [result_dir_, model_dir_]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

        
            net = skorch.NeuralNetRegressor(
                module=BasicLSTM(hidden_size,num_layers,num_features,device,dropout),
                criterion=nn.MSELoss,
                optimizer=torch.optim.Adam,
                device=device,
                batch_size=-1,
                train_split=None,
                max_epochs=1,
                warm_start=True,
                verbose=0,
            )
            best_net = net
            with open(model_dir + '/model_setting.pkl','wb') as out_model:
                pickle.dump(net, out_model)


            train_history = []
            best_loss = None
            for epoch in range(num_epochs):
                # start_time = time.time()
                history = {'epoch': epoch+1,
                          'best_model': False}
                train_loss, valid_loss = 0, 0

                # train and save average loss
                for i, train_batch in enumerate(DataLoader(train_set,batch_size=128,num_workers=6, pin_memory=True)):
                    X_train, y_train = train_batch[:, :-1].type(torch.float32), train_batch[:,1:].type(torch.float32)
                    net.fit(X_train, y_train)
                    train_loss += net.history[-1]['train_loss']
                train_loss /= i+1

                # validation & save average loss
                for i, train_batch in enumerate(DataLoader(valid_set,batch_size=128,num_workers=6,pin_memory=True)):
                    X_valid, y_valid = train_batch[:, :-1].type(torch.float32), train_batch[:,1:].type(torch.float32)
                    y_pred_valid = torch.Tensor(net.predict_proba(X_valid))
                    valid_loss += criterion(y_pred_valid, y_valid).detach().numpy()
                valid_loss /= i+1
                history['train_loss'] = train_loss
                history['valid_loss'] = valid_loss
                history['duration'] = time.time() - start_time

                # if valid loss best: save the model
                if best_loss is None or best_loss > valid_loss:
                    best_loss = valid_loss
                    history['best_model'] = True
                    net.save_params(f_params=model_dir_+'/model.pt', f_optimizer=model_dir_+'/optimizer.pt')
                train_history.append(history)
                print(train_history[-1])

            # save training history
            for last_epoch in train_history[::-1]:
                if last_epoch['best_model']:
                    print('Best model is achieved after {} epochs, validation loss = {:.4f}'
                          .format(last_epoch['epoch'], last_epoch['valid_loss']))
                    break
            with open(model_dir_+'/history.json', 'w') as output:
                json.dump(train_history, output)

         # initialize the best model and predict
            best_net.initialize()
            best_net.load_params(
                f_params=model_dir_+'/model.pt', f_optimizer=model_dir_+'/optimizer.pt'
            )

            for _, test_data in enumerate(DataLoader(test_set, batch_size=test_len)):
                y_pred = np.empty(test_data.numpy().shape)
                print('Testing')
                for i in range(len(y_pred)):
                    y_pred[i] = expanding_pred(best_net, test_data[i])
                    print('Testing progress: {}/{}'.format(i + 1, len(y_pred)), end='\r')
                print()

                if not os.path.exists(result_dir_):
                    os.makedirs(result_dir_)
                np.save(os.path.join(result_dir_, 'y_pred.npy'), np.float32(y_pred))
                np.save(os.path.join(result_dir_, 'y_test.npy'), test_data.numpy())
    
    elapsed_time = time.time() - start_time()
    print('time elapsed: ', elapsed_time)


if __name__ == '__main__':
    main()


# ## Plot Model

# In[ ]:


import numpy as np
import os
import matplotlib.pyplot as plt

# date='6-27-19'

session_name = '1000_epochs_adam'
result_dir = 'result/' + session_name + '/'

for k in range(10000,60000,10000):
    numBCName = str(k)
    result_dir_ = os.path.join(result_dir, numBCName)
    filename_pred = os.path.join(result_dir_, 'y_pred.npy')
    filename_test = os.path.join(result_dir_, 'y_test.npy')
    y_pred = np.load(filename_pred)
    y_test = np.load(filename_test)
    print(np.shape(y_pred))
    print(np.shape(y_test))

    test_i = 1500

    numPoints = 16

    D = y_pred[test_i][0][0]
    r = []
    interval = D/numPoints
    x = int(numPoints/2)
    for j in range(-x,x+1):
        r.append(interval*j)
    
    figNum = int(k/10000)
    plt.figure(num=figNum,dpi=600)
    for i in range(len(y_test[test_i])):
        if i != 0:
            plt.plot(y_test[test_i][i],r,label='t = %s' %(i))
    plt.title('True Velocity Profile Development for D = %fm' %(D))
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')

    plt.figure(num=figNum+5,dpi=600)
    for i in range(len(y_pred[test_i])):
        if i != 0:
            plt.plot(y_pred[test_i][i],r,label='t = %s' %(i))
    plt.title('Predicted Velocity Profile Development for D = %fm' %(D))
    plt.xlabel('velocity [m/s]')
    plt.ylabel('r [m]')
    plt.legend(bbox_to_anchor=(1,1),fontsize='x-small')


# ## Performance Metrics

# In[ ]:


import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

numBCs = 10000
minD = 0.50
maxD = 1.00
numTimeSteps = 20

session_name = '1000_epochs_adam'
result_dir = 'result/' + session_name + '/'

for i in range(10000,60000,10000):
    numBCName = str(i)
    result_dir_ = os.path.join(result_dir, numBCName)
    filename_pred = os.path.join(result_dir_, 'y_pred.npy')
    filename_test = os.path.join(result_dir_, 'y_test.npy')
    y_pred = np.load(filename_pred)
    y_test = np.load(filename_test)
    filename_text = os.path.join(result_dir_, 'performance metrics ' + numBCName + ' data points')

    num_samples = len(y_pred)

    y_pred = y_pred.reshape(num_samples * numTimeSteps, -1)
    y_test = y_test.reshape(y_pred.shape)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mse_text = 'mse: %.10f' %(mse)
    mae_text = 'mae: %.10f' %(mae)

    file = open(filename_text, 'w')
    file.write("Performance Metrics \n")
    file.write(mse_text +'\n')
    file.write(mae_text)
    file.close()

