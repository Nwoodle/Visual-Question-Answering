'''
The datalodaer for vqa
'''
import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as td
import torchvision.transforms as transforms

import config
import utils

class VQADataset(td.Dataset):

    def __init__(self, questions_path, answers_path, image_features_path, answerable_only=False, mode="train", image_size=(224, 224)):
        
        super(VQADataset, self).__init__()

        self.image_size = image_size
        self.mode = mode

        if mode == "train":
            self.images_dir = utils.path_for(train=True)
            self.data = #TODO
        if mode == "test":
            self.images_dir = utils.path_for(test=True)
            self.data = #TODO
        if mode == "val":
            self.images_dir = utils.path_for(val=True)
            self.data = #TODO




    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "VQADataset(mode={})". \
               format(self.mode)
    
    def __getitem__(self, idx):
        # TODO GET TUPLE OF DATA
        image_id = vqa_annotation[0]
        question = vqa_annotation[1]
        answer = vqa_annotation[2]
        img_path = os.path.join(self.images_dir, \
                                image_id)
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        v = transform(img)
        return v, question, answer

    def number_of_classes(self):
        return self.data['class'].max() + 1