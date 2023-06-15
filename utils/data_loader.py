import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.build_vocab import Vocabulary

root_dir='./data/coco'
train_annotations_file='./data/coco/annotations_DCC/captions_no_caption_rm_eightCluster_train2014.json'

def get_val_annotations_file(root,mode):
    if '_' in mode:
        category,mode=mode.split('_')
        file_name=f'captions_split_set_{category}_val_{mode}_novel2014.json'
        return os.path.join(root,'annotations_DCC/',file_name)
    else:
        file_name=f'captions_val_{mode}2014.json'
        return os.path.join(root,'annotations_DCC/',file_name)



class CoCoDataset(Dataset):
    def __init__(self,mode,transform,root=root_dir,max_len=25,small=False):
        self.mode = mode
        self.transform = transform
        self.max_len=max_len
        self.small=small
        self.vocab = Vocabulary()

        coco_type='train2014' if mode == 'train' else 'val2014'
        self.img_rdir = os.path.join(root, 'images/{}'.format(coco_type))
        #print(self.img_rdir)
        if mode == 'train':
            self.coco=COCO(train_annotations_file)
        else:
            self.coco=COCO(get_val_annotations_file(root,mode))
        self.ids = list(self.coco.anns.keys())            

    def __getitem__(self, index):        
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_dir=os.path.join(self.img_rdir,img_name)
        image= Image.open(img_dir).convert('RGB')
        image = self.transform(image)
        
        tokens = str(caption).lower().split()
        caption = [self.vocab(self.vocab.start_word)]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        caption = sentence_pad(caption,self.max_len)
        caption = torch.Tensor(caption).long()
        img_id=np.array(img_id)
        return image,caption,img_id

    def __len__(self):
        if self.small:
            return 1000
        else:
            return len(self.ids)


        
def sentence_pad(sentence, max_len):
    sentence = sentence[:max_len]
    if len(sentence) < max_len:
        sentence += [0] * (max_len - len(sentence))
    return sentence

def get_loader(mode='train',batch_size=128,num_workers=0,transform=None,max_len=25,small=False):
    
    dataset=CoCoDataset(mode=mode,transform=transform,max_len=max_len,small=small)

    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)

    return data_loader


