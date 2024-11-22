# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 08:57:47 2021

@author: Xiaonan Liu
"""

# Import packages
import os
import random
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Building model 指定随机种子
def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

class Linear_ANN(nn.Module):
    """
        Layer of our ANN. 定义一个全连接层
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize the weights and the bias
        self.weight = nn.Parameter(torch.randn(output_features, input_features)*0.01)
        self.bias = nn.Parameter(torch.zeros(output_features))      

    def forward(self, input):
        """
          Optimization process
        """
        return func.linear(input, self.weight, self.bias)
    
class Neural3network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim, p=0.2):
        # call constructor from superclass
        super(Neural3network, self).__init__()
        
        # define network layers
        self.layer1 = Linear_ANN(in_dim, n_hidden_1) #定义第一层线性网络
        self.layer2 = Linear_ANN(n_hidden_1, out_dim) #定义第二层线性网络
        
        self.dropout = nn.Dropout(p)  # dropout训练

    def forward(self, x):
        # define forward pass
        x = x.view(x.size(0), -1) #把矩阵转为一维张量
        x = self.dropout(self.layer1(x))  #第一层线性层，dropout随机丢掉神经元
        x = func.relu(x)       #指定第一层的激活函数为relu
        x = torch.sigmoid(self.layer2(x))  #指定第二层的激活函数为sigmiod。
        return x

#Read data
class MyDataset(Dataset):
    """
    定义pytorch里的dataset类
    """
    def __init__(self, col=1):
        data1 = np.loadtxt('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/Environment_select.csv',delimiter=',',skiprows=1,usecols=range(1,46), dtype=np.float32)
        data2 = np.loadtxt('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/Microbiom-ASV-alpha_select.csv',delimiter=',',skiprows=1,usecols=col, dtype=np.float32)  #预测 alpha多样性
        #data2 = np.loadtxt('/media/wxl/Run1/lxn/WWTP_ML/rawdata/Microbiom-ASV-10_select.csv',delimiter=',',skiprows=1,usecols=col, dtype=np.float32)   预测 >10% ASV的丰度
        #data2 = np.loadtxt('/media/wxl/Run1/lxn/WWTP_ML/rawdata/ASV_select-MiDAS-functioner.csv',delimiter=',',skiprows=1,usecols=col, dtype=np.float32)    预测 Midas-function
        ##Normalization
        data2_normed = (data2 - data2.min(axis=0) + 1e-12)/(data2.max(axis=0)-data2.min(axis=0) + 1e-12)  #对data2进行标准化处理
        
        state = np.random.get_state()
        np.random.shuffle(data1)  #随机打乱数据
        np.random.set_state(state)
        np.random.shuffle(data2_normed)
        
        self.features = torch.from_numpy(data1)
        self.targets = torch.reshape(torch.from_numpy(data2_normed),(772,1))
        self.len = data1.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.features[idx],self.targets[idx]
    

def get_kfold_data(k, i, data,test_pct=0):  
    test_len = int(data.len*test_pct)
    test_idx= list(range(data.len-test_len,data.len))
    X_test = data.__getitem__(test_idx)[0]
    y_test = data.__getitem__(test_idx)[1]
    
    train_idx = list(range(data.len-test_len))
    X = data.__getitem__(train_idx)[0]
    y = data.__getitem__(train_idx)[1]
    fold_size = X.shape[0] // k
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:
        X_valid, y_valid = X[val_start:], y[val_start:] 
        X_train = X[0:val_start]
        y_train = y[0:val_start]
        
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def traink(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, wd, TOTAL_EPOCHS):
    def _init_fn(worker_id):
        random.seed(int(0) + worker_id)

    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()  # loss function
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, betas=(0.9, 0.999), weight_decay=wd)

    losses = []
    train_losses = []
    val_losses = []
    train_R2 = []
    val_R2 = []

    for epoch in range(TOTAL_EPOCHS):
        y_Train = []
        y_Val = []
        y_Pred = []
        y_Pred_ = []

        # Training
        model.train()
        for i, (x_data, y_data) in enumerate(train_loader):
            real_cpu, label_cpu = x_data, y_data
            if torch.cuda.is_available():
                real_cpu = real_cpu.cuda()
                label_cpu = label_cpu.cuda()
            inputs = real_cpu
            labels = label_cpu

            optimizer.zero_grad()
            inputv = Variable(inputs)
            labelv = Variable(labels)

            outputs = model(inputv)
            loss = criterion(outputs, labelv)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            y_Train.append(labelv)
            y_Pred.append(outputs.data)

        train_losses.append(loss.item())
        R2 = r2_score([t.cpu().numpy() for t in y_Train][0], [t.cpu().numpy() for t in y_Pred][0], sample_weight=None,
                      multioutput='uniform_average')
        train_R2.append(R2.item())

        # Validation
        model.eval()
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                real_cpu, label_cpu = x_val, y_val
                if torch.cuda.is_available():
                    real_cpu = real_cpu.cuda()
                    label_cpu = label_cpu.cuda()
                inputs_val = real_cpu
                labels_val = label_cpu

                inputv_val = Variable(inputs_val)
                labelv_val = Variable(labels_val)
                outputs_val = model(inputv_val)

                val_loss = criterion(outputs_val, labelv_val)
                y_Val.append(labelv_val)
                y_Pred_.append(outputs_val.data)

        val_losses.append(val_loss.item())
        R2_val = r2_score([t.cpu().numpy() for t in y_Val][0], [t.cpu().numpy() for t in y_Pred_][0],
                          sample_weight=None, multioutput='uniform_average')
        val_R2.append(R2_val.item())

        # **Print the loss for each epoch**
        print(f'Epoch [{epoch + 1}/{TOTAL_EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses, train_R2, val_R2


def k_fold(infor_file, k, mydata, seed, drop, num_epochs = 100, batch_size = 16, learning_rate = 0.00001, weight_decay=0):
    
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum , valid_acc_sum = 0, 0
    tra_loss_sum = np.zeros(num_epochs)
    tra_acc_sum = np.zeros(num_epochs)
    val_loss_sum = np.zeros(num_epochs)
    val_acc_sum = np.zeros(num_epochs)
    
    print('Seed:{:d}, Wd:{:.4f}, Drop_p:{:.2f}\n'.format(seed, weight_decay, drop), file=infor_file)
    
    for i in range(k):
        print('The', i + 1,'fold', file=infor_file)

        data = get_kfold_data(k, i, mydata,0.2) 
        model = Neural3network(45, 91, 1, drop)
        train_loss, val_loss, train_R2, val_R2 = traink(model, data[0], data[1], data[2], data[3], batch_size, learning_rate, weight_decay, num_epochs) 
       
        print('train_loss(MSE+L2):{:.5f}, train_R2:{:.3f}'.format(train_loss[-1], train_R2[-1]),file=infor_file)
        print('valid loss(MSE+L2):{:.5f}, valid_R2:{:.3f}\n'.format(val_loss[-1], val_R2[-1]),file=infor_file)
        
        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_acc_sum += train_R2[-1]
        valid_acc_sum += val_R2[-1]
        tra_loss_sum += np.array(train_loss)
        val_loss_sum += np.array(val_loss)
        tra_acc_sum += np.array(train_R2)
        val_acc_sum += np.array(val_R2)
        
    print('\n', '#'*k,'最终4折交叉验证结果','#'*k,file=infor_file) 
    print('average train loss:{:.4f}, average train accuracy:{:.3f}'.format(train_loss_sum/k, train_acc_sum/k),file=infor_file)
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}'.format(valid_loss_sum/k, valid_acc_sum/k),file=infor_file)
    
    return model,data

os.chdir('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/results/alpha/') #change direction.
os.getcwd() #get current work direction.

for col in range(1,5):
    for seed in range(0,20):   
        for wd in [0.01]:
            drop_P = 0.4
            #固定种子
            seed_torch(seed)
            #加载训练数据
            mydata = MyDataset(col)
            filename = str(col) + 'col-4fold-seed'+ str(seed) +'-10000ep-256bS-0.00001lr-'+ str(wd) +'wd-drop'+ str(drop_P)+'.txt'
            infor_file = open(filename, 'w+')
            #K交叉验证
            Net, Set = k_fold(infor_file,4,mydata,seed,drop_P,10000,78,0.00001,wd)
            infor_file.close()
            pth_name = str(col) + 'col-4fold-seed'+ str(seed) +'-10000ep-78bS-0.00001lr-'+ str(wd) +'wd-drop'+ str(drop_P) +'_train_network.pth'
            torch.save(Net, pth_name)    #保存训练好的网络            
print('Finished the training process!')
