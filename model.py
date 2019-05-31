'''
The model for VQA
'''
import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv
import nntools as nt

class NNClassifier(nt.NeuralNetwork):

    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def criterion(self, y, d):
        return self.cross_entropy(y, d)

class VQANet(NNClassifier):

    def __init__(self, vocab_size, embedding=True, lstmdim=512):
        '''
        Args:
            vocab_size: all the words used in the dictionary
            # target_size: output vector size
            embedding: False if the input sentence is already embedded
            lstmdim: the dim of the lstm
        '''
        super(VQANet, self).__init__()
        # Question Channel
        # Input Embedding
        if embedding:
            self.word_embeddings = nn.Embedding(vocab_size, lstmdim)
            
        # Question channel: LSTM for 512*512 with 1 hidden layer
        self.lstm = nn.LSTM(lstmdim, lstmdim, batch_first=True)
        self.lstmoutput = nn.Linear(lstmdim,1024)
        
        # Image Channel
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
        self.imagefeatures = vgg.features
        self.imageclassifier = vgg.classifier
        num_ftrs = vgg.classifier[6].in_features
        self.imageclassifier[6] = nn.Linear(num_ftrs, 1024)
        # self.vggout = nn.Linear(num_ftrs, 1024)

        # Output Channel
        self.combinefc = nn.Linear(1024,1000)

    # def init_hidden(self):

    def forward(self, q, v):
        '''
        Args:
            q: question tensor list with n diemention tensors
            v: image
        Return:
            y: vector of answer
        Note:
            In the pytoch documentation, the lstm input is reshaped using view(len(sentence), 1, -1) 
        '''
        # Question Channel
        # if embedding:
        #     embeds = self.word_embeddings(q)
        # else:
        #     embeds = q
        # embeds = q[0,:,:]
        embeds = q
        # v = v[0,:,:]
        _, (_, lstmout) = self.lstm(embeds)
        lstmout = lstmout.squeeze(0)
        # lstmout, _ = self.lstm(embeds.view(len(embeds), 1, -1))
        lstmout = self.lstmoutput(lstmout)
        # lstmout = lstmout.view(lstmout.size(0),-1)
        lstmout = F.tanh(lstmout)
        
        # Image Channel
        imageout = self.imagefeatures(v)
        imageout = imageout.view(imageout.size(0), -1)
        imageout = self.imageclassifier(imageout)
        # imageout = self.vggout(imageout)

        # Combine two channel together
        combineout = lstmout * imageout
        combineout = self.combinefc(combineout)
        y = F.softmax(combineout, dim=0)

        return y

class VQAStatsManager(nt.StatsManager):

    def __init__(self):
        super(VQAStatsManager, self).__init__()

    def init(self):
        super(VQAStatsManager, self).init()
        self.running_accuracy = 0

    def accumulate(self, loss, y, d):
        super(VQAStatsManager, self).accumulate(loss, y, d)
        _, l = torch.max(y, 1)
        self.running_accuracy += torch.mean((l == d).float())

    def summarize(self):
        loss = super(VQAStatsManager, self).summarize()
        accuracy = 100 * self.running_accuracy / self.number_update
        return {'loss': loss, 'accuracy': accuracy}