import json
import os
import pickle
import random
from collections import Counter
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from skimage import io
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as F

class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file,annotations_file,vocab_from_file):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file,annotations_file, vocab_from_file)
        # self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens= []
            for id in tqdm(self.ids):
                caption = str(self.coco.anns[id]['caption'])
                tokens = caption.split()
                all_tokens.append(tokens)
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            # self.paths = [item['file_name'] for item in test_info['images']]
            self.paths = [item['url'] for item in test_info['images']]

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            img = self.coco.loadImgs(img_id)[0]
            img_dir='./data/images/train2014/'+img['file_name']

            image= Image.open(img_dir).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = str(caption).lower().split()
            caption = [self.vocab(self.vocab.start_word)]
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            # PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            # orig_image = np.array(PIL_image)
            # image = self.transform(PIL_image)
            orig_image = io.imread(path)
            image = self.transform(Image.fromarray(orig_image))
            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TORCH_HOME'] = os.path.join('./data', 'pretrained_models')
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_loader(transform=None,
               mode='train',
               batch_size=128,
               vocab_threshold=None,
               vocab_file='./data/vocab.pkl',
               vocab_from_file=True,
               num_workers=0,
               data_loc='./data/annotations_DCC/',
               category=None,):
    
    assert mode in ['train', 'val', 'test'], "mode must be one of 'train' 'val' or 'test'."
    if not vocab_from_file:
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'train':
        transform = transform_train
        if vocab_from_file:
            assert os.path.exists(
                vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        # img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
        annotations_file = os.path.join(data_loc, 'captions_no_caption_rm_eightCluster_train2014.json')
    else:
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file, "Change vocab_from_file to True."
        assert category, "Must provide category name if in 'val' or 'test' mode."
        # img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
        # annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')
        annotations_file = os.path.join(data_loc, f'captions_split_set_{category}_val_{mode}_novel2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file)
    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
    return data_loader


transform_train = transforms.Compose([
    # use lanczos to get better results
    #transforms.Resize(256, interpolation=F._interpolation_modes_from_int(1)),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
