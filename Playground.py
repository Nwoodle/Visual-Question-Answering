#%%


import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import nntools as nt
import json
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

    def __init__(self, vocab_size, target_size, embedding=True, lstmdim=512):
        '''
        Args:
            vocab_size: all the words used in the dictionary
            target_size: output vector size
            embedding: False if the input sentence is already embedded
            lstmdim: the dim of the lstm
        '''
        super(VQANet, self).__init__()
        # Input Embedding
        if embedding:
            self.word_embeddings = nn.Embedding(vocab_size)
            
        # Question channel: LSTM for 512*512 with 1 hidden layer
        self.lstm = nn.LSTM(lstmdim, lstmdim)
        self.lstmoutput = nn.Linear(512,1024)


    def forward(self, q, v):
        '''
        Args:
            q: question tensor list with n 300 diemention tensors
            v: image
        Return:
            y: vector of answer
        Note:
            In the pytoch documentation, the lstm input is reshaped using view(len(sentence), 1, -1) 
        '''



        return y

#%%
embeded = nn.Embedding(6, 3)
lstminput = embeded(torch.tensor([[1,1],[3,2]]))
print(lstminput)
#%%
lstminput.view(2,1,-1).size()


#%%
with open("train_qna.json", 'r') as fd:
    qna = json.load(fd)
len(qna)

#%%
with open("vocab.json", 'r') as fd:
    vocab = json.load(fd)
vocab['answer']['yes']

#%%
torch.randn(1,2,3).size()

#%%
