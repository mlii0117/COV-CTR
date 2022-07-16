import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import os
import pickle as pkl
import json
import numpy as np
import sys


class Fh21Dataset(Dataset):
    def __init__(self, opt):
        
        # TODO
        self.data_dir = '/home/mmvg/Desktop/COVID/reports/processed_fh21_precise_tag/'
        self.num_medterm = opt.num_medterm

        with open(os.path.join(self.data_dir, 'fh21.pkl'), 'rb') as f:
            self.data = pkl.load(f)

        with open(os.path.join(self.data_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)

        self.idw2word = {v: k for k, v in self.word2idw.items()}

        self.vocab_size = len(self.word2idw)

    def __getitem__(self, index):
        ix = self.data[index][0]
        abstracts = self.data[index][1]
        abstracts = np.array(abstracts)

        abstracts_labels = self.data[index][2]
        abstracts_labels = np.array(abstracts_labels)

        abstracts = torch.from_numpy(abstracts).long()
        abstracts_labels = torch.from_numpy(abstracts_labels).long()

        medterm_labels = np.zeros(self.num_medterm)
        medterms = self.data[index][3]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                medterm_labels[medterm] = 1

        return ix, abstracts, abstracts_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        return len(self.data)


def get_loader2(opt):

    dataset = Fh21Dataset(opt)
    loader = DataLoader(dataset=dataset, batch_size=opt.train_batch_size,
                            shuffle=True, num_workers=16)
    return dataset, loader

