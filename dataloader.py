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

class VQADataset(td.Dataset):

    def __init__(self, mode="train", image_size=(224, 224), answer_num=1000):
        
        super(VQADataset, self).__init__()

        self.image_size = image_size
        self.mode = mode
        self.answer_num = answer_num

        if mode == "train":
            self.images_dir = os.path.join('mscoco', 'train2014')
            self.data_dir = "train_qna.json"
            self.imageprefix = 'COCO_train2014_'
        if mode == "test":
            self.images_dir = os.path.join('mscoco', 'test2015')
            self.imageprefix = 'COCO_test2015_'
            # self.data = #TODO
        if mode == "val":
            self.images_dir = os.path.join('mscoco', 'val2014')
            self.imageprefix = 'COCO_val2014_'
            # self.data = #TODO
        
        with open(self.data_dir, 'r') as fd:
            self.data = json.load(fd)
        self.maxqlen = 0
        for annotation in self.data:
            if len(annotation[2])>self.maxqlen:
                self.maxqlen = len(annotation[2])
        with open("vocab.json", 'r') as fd:
            self.vocab = json.load(fd)



    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "VQADataset(mode={})". \
               format(self.mode)
    
    def __getitem__(self, idx):
        image_id = str(self.data[idx][0])
        image_name = self.imageprefix + image_id.zfill(12) + '.jpg'
        question_word = self.data[idx][2]
        answer_word = self.data[idx][3]
        qlen = len(question_word)
        question = []
        answer = torch.zeros(self.answer_num)
        for word in question_word:
            question.append(self.vocab['question'][word])
        question = torch.tensor(question)
        word_embeddings = nn.Embedding(len(self.vocab['question']), 512)
        question = word_embeddings(question)
        question_padding = torch.zeros([self.maxqlen-qlen,512])
        question = torch.cat((question_padding,question))
        try:
            answeridx = self.vocab['answer'][answer_word]
        except:
            answeridx = -1
        if answeridx != -1:
            answer[answeridx] = 1
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
