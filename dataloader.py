'''
The datalodaer for vqa
'''
import torch.utils.data as td

class VQADataset(td.Datasset):

    def __init__(self, mode="train", image_size=(224, 224)):
        super(VQADataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        if mode == "train":
            self.images_dir = os.path.join("mscoco", "train2014")
            self.data = #TODO
        if mode == "test":
            self.images_dir = os.path.join("mscoco", "test2015")
            self.data = #TODO
        if mode == "val":
            self.images_dir = os.path.join("mscoco", "val2014")
            self.data = #TODO

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "VQADataset(mode={})". \
               format(self.mode)
    
    def __getitem__(self, idx):
        #TODO
        img_path = os.path.join(self.images_dir, \
                                self.data.iloc[idx]['file_path'])
        bbox = self.data.iloc[idx][['x1', 'y1', 'x2', 'y2']]
        img = Image.open(img_path).convert('RGB')
        img = img.crop([bbox[0], bbox[1], bbox[2], bbox[3]])
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        x = transform(img)
        d = self.data.iloc[idx]['class']
        return x, d

    def number_of_classes(self):
        return self.data['class'].max() + 1


    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, \
                                self.data.iloc[idx]['file_path'])
        bbox = self.data.iloc[idx][['x1', 'y1', 'x2', 'y2']]
        img = Image.open(img_path).convert('RGB')
        img = img.crop([bbox[0], bbox[1], bbox[2], bbox[3]])
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        x = transform(img)
        d = self.data.iloc[idx]['class']
        return x, d

    def number_of_classes(self):
        return self.data['class'].max() + 1        
