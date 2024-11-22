# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 22:41:28 2022

@author: Xiaonan Liu
"""
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as func
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


# 封装种子设置函数
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Linear_ANN(nn.Module):
    """
        Layer of our ANN.
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
        self.weight = nn.Parameter(torch.randn(output_features, input_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, input):
        """
          Optimization process
        """
        return func.linear(input, self.weight, self.bias)


class Neural3network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim, p=0):
        # call constructor from superclass
        super(Neural3network, self).__init__()

        # define network layers
        self.layer1 = Linear_ANN(in_dim, n_hidden_1)
        self.layer2 = Linear_ANN(n_hidden_1, out_dim)

        self.dropout = nn.Dropout(p)  # dropout训练

    def forward(self, x):
        # define forward pass
        x = x.view(x.size(0), -1)
        x = self.dropout(self.layer1(x))
        x = func.relu(x)
        x = torch.sigmoid(self.layer2(x))
        # x = func.relu(self.layer2(x))
        return x


class MyDataset(Dataset):
    def __init__(self, col=1):
        data1 = np.loadtxt('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/Environment_select_DIS_5.csv', delimiter=',', skiprows=1,
                           usecols=range(1, 46), dtype=np.float32)
        data2 = np.loadtxt('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/Microbiom-ASV-alpha_select.csv', delimiter=',',
                           skiprows=1, usecols=col, dtype=np.float32)

        data2_normed = (data2 - data2.min(axis=0) + 1e-12) / (data2.max(axis=0) - data2.min(axis=0) + 1e-12)

        state = np.random.get_state()
        indices = np.arange(data1.shape[0])  # 保存原始索引
        np.random.shuffle(data1)
        np.random.set_state(state)
        np.random.shuffle(data2_normed)
        np.random.shuffle(indices)  # 与数据同步打乱索引

        self.features = torch.from_numpy(data1)
        self.targets = torch.reshape(torch.from_numpy(data2_normed), (772, 1))
        self.length = data1.shape[0]
        self.indices = torch.from_numpy(indices)  # 将索引保存为属性

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.indices[idx]  # 返回特征、目标和原始索引

def get_testdata(col=1):
    data = MyDataset(col)
    return data.features, data.targets, data.indices  # 返回特征、目标和索引


os.chdir('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/results/alpha/')  # change direction.
os.getcwd()  # get current work direction.

para = np.ones((0, 52))
Test_result = np.ones((0, 52))

for col in range(1, 4):
    for seed in range(0, 20):
        drop_P = 0
        wd = 0.01
        # 加载模型
        net = torch.load(
            str(col) + 'col-4fold-seed' + str(seed) + '-10000ep-256bS-0.00001lr-' + str(wd) + 'wd-drop' + str(
                drop_P) + '_train_network.pth', map_location='cpu')
        # 计算参数最终的权重(重要性)和偏移量(Garson’s Algorithm)
        weight_H_To_O = np.diag(net.layer2.weight.data.numpy()[-1])
        weight_I_To_H = net.layer1.weight.data.numpy()
        weight_final = np.dot(weight_H_To_O, weight_I_To_H)
        abs_weight_final = abs(weight_final)
        weightForH = abs_weight_final / abs_weight_final.sum(axis=1, keepdims=True)  ##每行除以行和，得到经过一个隐节点，每个输入节点的相对贡献
        weightForH[np.isnan(weightForH)] = 0  ##有drop或某些隐节点对输出权重为0时，出现nan
        Sum_input = weightForH.sum(axis=0)
        RI = Sum_input / Sum_input.sum()  ##每个节点的相对重要性
        para_bias = np.dot(net.layer2.weight.data.numpy(), net.layer1.bias.data.numpy()) + net.layer2.bias.data.numpy()
        # 固定种子
        seed_torch(seed)
        ##加载test数据
        x_test, y_test, indices = get_testdata(col)  # 获取样本索引
        test_len = x_test.shape[0]
        col_result = np.ones((test_len, 1)) * col
        seed_result = np.ones((test_len, 1)) * seed
        wd_result = np.ones((test_len, 1)) * wd
        drop_P_result = np.ones((test_len, 1)) * drop_P
        ##测试模型
        net.eval()
        with torch.no_grad():
            Test_pred = net(x_test)
            Test_pred_ = Test_pred.data.numpy()
            y_test_ = y_test.numpy()
            x_test_ = x_test.numpy()
            # 计算模型测试结果
            MSE_T = mean_squared_error(y_test_, Test_pred_, sample_weight=None, multioutput='uniform_average')
            R2_T = r2_score(y_test_, Test_pred_)

        # 整合结果
        each_test = np.c_[indices, col_result, seed_result, wd_result, drop_P_result, x_test_, y_test_, Test_pred_]  # 包含索引
        Test_result = np.r_[Test_result, each_test]

        each = np.c_[col, seed, wd, drop_P, MSE_T, R2_T, RI.reshape(1, 45), para_bias]
        para = np.r_[para, each]

np.savetxt('/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/results/alpha/analysis新45/' + 'Test_log-10_select-sigmoid.txt',
           Test_result, fmt='%.4e', delimiter='\t', newline='\n')
np.savetxt(
    '/home/hongchang/Documents/pytorch计算/WWTPs温度扰动课题/results/alpha/analysis新45/' + 'Test_and_parameters-10_select-sigmoid.csv',
    para, fmt='%.4e', delimiter=',', newline='\n')
