#!/usr/bin/env python
# coding: utf-8

#%%


import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import nntools as nt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#%%
class NNClassifier(nt.NeuralNetwork):

    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def criterion(self, y, d):
        return self.cross_entropy(y, d)


#%%
class VQANet(NNClassifier):

    def __init__(self, D, C=64):
        super(VQANet, self).__init__()
        '''
        '''
        # 300 word embedding to 512 lstm imput
        self.lstminput = nn.Linear(300,512)
        # Question channel: LSTM for 512*512 with 1 hidden layer
        self.lstm = nn.LSTM(512,512)
        self.lstmoutput = nn.Linear(512,1024)


    def forward(self, q, v):
        '''
        Args:
            q: question list with n 300 diemention tensors
            v: image
        Return:
            y: vector of answer
        '''


        return y

#%%
torch.LongTensor([[0,2,0,5]])
#%%
