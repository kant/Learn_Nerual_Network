# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:50:27 2020

@author: Xu Mingjun
"""

import torch
import torch.nn as nn
import numpy as np
from openpyxl import load_workbook
import torch.nn.functional as F

#读取表格中的数据
workbook = load_workbook('dataset\pendigits.xlsx')
booksheet = workbook.get_sheet_by_name('Sheet1')

#获取sheet页的行数据
rows = booksheet.rows
#获取sheet页的列数据
columns = booksheet.columns

# 迭代所有的行
data = np.zeros((10992,17))
feature = np.zeros((10992,16))
label = np.zeros((10992))

#把数据分为训练集和测试集分别是45和45
for i in range(10992):    
    for j in range(17): 
        data[i][j] = booksheet.cell(row=i+2, column=j+1).value

#默认对Ndarray的第一维打乱
#np.random.shuffle(data)  
      
#训练数据集
train_label = data[0:10000,-1]     #[0:50]不包含50
train_feature = data[0:10000,0:-1]
#测试数据集
test_label = data[10000:10992,-1]
test_feature = data[10000:10992,0:-1]

#对label进行onehot编码
train_label_oh = np.eye(10)[train_label.astype(int)]
test_label_oh = np.eye(10)[test_label.astype(int)]

#对feature归一化处理
train_feature = train_feature / train_feature.max(axis=0)
test_feature = test_feature / test_feature.max(axis=0)

#训练和测试进行类型转换
train_feature_t = torch.from_numpy(train_feature)
train_label_t = torch.from_numpy(train_label_oh)

train_feature_t_f = torch.tensor(train_feature_t,dtype=torch.float32)
train_label_t_f = torch.tensor(train_label_t,dtype=torch.float32)

test_feature_t = torch.from_numpy(test_feature)
test_label_t = torch.from_numpy(test_label_oh)

test_feature_t_f = torch.tensor(test_feature_t,dtype=torch.float32)
test_label_t_f = torch.tensor(test_label_t,dtype=torch.float32)

#######################训练过程##############################

#训练数据个数，输入维度，隐藏层神经元个数，输出维度
N, D_in, H1, H2, D_out = 10000, 16, 20, 15, 10
H3, H4 = 20, 10

class ADNet(torch.nn.Module):
    #构建三层全连接网络
    def __init__(self, D_in, H1, H2, D_out):
        super(ADNet, self).__init__()
        # define the model architecture
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
        
    def forward(self, x):
        y_1 = self.linear1(x).clamp(min=0);
        y_2 = self.linear2(y_1).clamp(min=0);
        y_pred = self.linear3(y_2);
        return y_pred

class DADNet(torch.nn.Module):
    #构建三层全连接网络
    def __init__(self, D_in, H1, H2, H3, H4, D_out):
        super(DADNet, self).__init__()
        # define the model architecture
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, H4)
        self.linear5 = torch.nn.Linear(H4, D_out)
        
    def forward(self, x):
        y_1 = F.relu(self.linear1(x))
        y_2 = F.relu(self.linear2(y_1))
        y_3 = F.relu(self.linear3(y_2))
        y_4 = F.relu(self.linear4(y_3))
        y_pred = self.linear5(y_4)
        return y_pred   
    
model = ADNet(D_in, H1, H2, D_out)
#model = DADNet(D_in, H1, H2, H3, H4, D_out)
loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(20000): 
    # if t < 4000:
    #     learning_rate = 10;
    # elif t >= 4000 and t< 10000:
    #     learning_rate = 0.1;
    # elif t >= 10000:
    #     learning_rate = 0.01;
    y_pred = model(train_feature_t_f)  # model.forward()
    
    # compute loss
    loss = loss_fn(y_pred, train_label_t_f)
    print("训练次数：",t,"\tloss =",loss.item())
    
    optimizer.zero_grad()
    #Backward pass
    loss.backward()
    
    optimizer.step()

model.eval()

cnt = 0

test_out = model(test_feature_t_f)

for test_i in range(992):
    _, test_out_np= torch.max(test_out[test_i],0)

    print("No.",test_i,"\npre:",test_out_np.numpy(),"\nGT:",test_label[test_i])
    print("****************")
    if test_out_np == test_label[test_i]:
        #print("correct")
        cnt += 1;
    
correct_rate = cnt/992.0;
print("correct_rate:",correct_rate)