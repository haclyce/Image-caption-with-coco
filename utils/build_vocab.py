from pycocotools.coco import COCO
import pickle
import os
from tqdm import tqdm
from collections import Counter

class Vocabulary(object):
    def __init__(self,
                 vocab_threshold=5,
                 vocab_file='./vocab/vocab.pkl',
                 annotations_file='./data/coco/annotations/captions_train2014.json'
                ):
        
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.padding_word = "<pad>"
        self.start_word = "<start>"
        self.end_word = "<end>"
        self.unk_word = "<unk>"
        self.annotations_file = annotations_file
        # self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            #print('Vocabulary successfully loaded!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):

        self.init_vocab()
        self.add_word(self.padding_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        # print("Building vocabulary from captions...")
        for id in tqdm(ids):
            caption = str(coco.anns[id]['caption']).lower()
            tokens = caption.split()
            counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for word in words:
            self.add_word(word)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)