import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image

import os
import pickle as pkl
import json
import numpy as np
import sys


class CHXrayDataSet2(Dataset):
    def __init__(self, opt, split, transform=None):
        self.transform = transform

        self.data_dir = opt.data_dir
        # TODO
        self.pkl_dir = os.path.join('/home/mmvg/Desktop/COVID', 'reports')
        self.img_dir = os.path.join(self.data_dir, 'NLMCXR_png')

        self.num_medterm = opt.num_medterm

        with open(os.path.join(self.pkl_dir, 'align2.' + split + '.pkl'), 'rb') as f:
            self.findings = pkl.load(f)
            self.findings_labels = pkl.load(f)
            self.image = pkl.load(f)
            self.medterms = pkl.load(f)

        f.close()

        with open(os.path.join(self.pkl_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)
        f.close()

        with open(os.path.join(self.pkl_dir, 'idw2word.pkl'), 'rb') as f:
            self.idw2word = pkl.load(f)
        f.close()

        self.ids = list(self.image.keys())
        self.vocab_size = len(self.word2idw)

    def __getitem__(self, index):
        ix = self.ids[index]
        image_id = self.image[ix]
        image_name = os.path.join(self.img_dir, image_id)
        img = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        #print(img.size(), image_id)
        medterm_labels = np.zeros(self.num_medterm)
        medterms = self.medterms[ix]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                medterm_labels[medterm] = 1

        findings = self.findings[ix]
        findings_labels = self.findings_labels[ix]
        findings = np.array(findings)
        findings_labels = np.array(findings_labels)

        findings = torch.from_numpy(findings).long()
        findings_labels = torch.from_numpy(findings_labels).long()


        return ix, image_id, img, findings, findings_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        return len(self.ids)


def get_loader_cn(opt, split):

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    dataset = CHXrayDataSet2(opt, split=split,
                             transform=transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize
                             ]))
    if split == 'train':
        loader = DataLoader(dataset=dataset, batch_size=opt.train_batch_size,
                            shuffle=True, num_workers=16)
    elif split == 'val':
        loader = DataLoader(dataset=dataset, batch_size=opt.eval_batch_size,
                            shuffle=True, num_workers=16)
    elif split == 'test':
        loader = DataLoader(dataset=dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=16)
    else:
        raise Exception('DataLoader split must be train or val.')
    return dataset, loader

