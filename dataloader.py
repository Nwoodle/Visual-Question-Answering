'''
The datalodaer for vqa
'''
import json
import os
import os.path
import re
from torch import nn

from PIL import Image
import h5py
import torch
import torch.utils.data as td
import torchvision.transforms as transforms

import config
import utils
import copy

class VQADataset(td.Dataset):

    def __init__(self, mode="train", image_size=(224, 224), answer_num=1000):
        
        super(VQADataset, self).__init__()

        self.image_size = image_size
        self.mode = mode
        self.answer_num = answer_num

        if mode == "train":
            self.images_dir = os.path.join('C:\\Users\\johns\\285proj\\dataset', 'train2014')
            self.data_dir = "train_qna.json"
            self.imageprefix = 'COCO_train2014_'
        if mode == "test":
            self.images_dir = os.path.join('C:\\Users\\johns\\285proj\\dataset', 'test2015')
            self.imageprefix = 'COCO_test2015_'
            self.data_dir = "val_qna.json"
        if mode == "val":
            self.images_dir = os.path.join('C:\\Users\\johns\\285proj\\dataset', 'val2014')
            self.imageprefix = 'COCO_val2014_'
            # self.data = #TODO
        
        with open(self.data_dir, 'r') as fd:
            self.data = json.load(fd)
        self.data = self.data[1:4096]
        self.maxqlen = 0
        for annotation in self.data:
            if len(annotation[2])>self.maxqlen:
                self.maxqlen = len(annotation[2])
        with open("vocab.json", 'r') as fd:
            self.vocab = json.load(fd)
        for qadata in self.data:
            question = copy.copy(qadata[2])
            qadata[2] = []
            for qword in question:
                try:
                    qadata[2].append(self.vocab['question'][qword])
                except:
                    qadata[2].append(0)
            try:
                qadata[3] = self.vocab['answer'][qadata[3]]
            except:
                qadata[3] = 0


    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "VQADataset(mode={})". \
               format(self.mode)
    
    def __getitem__(self, idx):
        image_id = str(self.data[idx][0])
        image_name = self.imageprefix + image_id.zfill(12) + '.jpg'
        # question_word = self.data[idx][2]
        # answer_word = self.data[idx][3]
        # qlen = len(question_word)
        qlen = len(self.data[idx][2])
        # question = []
        # for word in question_word:
        #     question.append(self.vocab['question'][word])
        question = self.data[idx][2]
        question = torch.tensor(question)
        answer = self.data[idx][3]
        word_embeddings = nn.Embedding(len(self.vocab['question']), 512)
        question = word_embeddings(question)
        question_padding = torch.zeros([self.maxqlen-qlen,512])
        question = torch.cat((question_padding,question))
        # try:
        #     answer = self.vocab['answer'][answer_word]
        # except:
        #     answer = -1
        answer = torch.tensor(answer)
        img_path = os.path.join(self.images_dir, image_name)
        img = Image.open(img_path)
        img = img.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        v = transform(img)
        # print(v.size())
        return v, question, answer
    
    def get_vocabsize(self):
        return len(self.vocab['question'])

def colloate_fn(data):
    '''
    Used for handling different length of question [n x 512]
    '''
