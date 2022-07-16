#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True


def decode_transformer_findings(idw2word, sampled_findings):
    decode_list = []
    n_samples, n_words = sampled_findings.size()

    for n in range(n_samples):
        decoded = []
        words = []
        for i in range(n_words):
            token = idw2word[int(sampled_findings[n][i])]
            if token == '<BOS>':
                continue
            if token == '<EOS>':
                break
            if token != '<UNK>' and token != '<BLANK>':
                words.append(token)
        if len(words) != 0:
            decoded.append(' '.join(words))
        decode_list.append(' '.join(decoded))
    return decode_list  # [batch_size, length]


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target):
        # truncate to the same size
        # target = target[:, :input.size(1)]
        # mask = mask[:, :input.size(1)]
        batch = input.size(0)
        length = input.size(1)
        epsion = 1e-8

        mask = (target == -1)
        target[mask] = 0
        mask = (1 - mask).float()

        input = F.log_softmax(input, dim=-1)
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        position_ids = torch.arange(start=1, end=length+1, dtype=torch.float, device=input.device).flip(0)
        position_ids = torch.pow(position_ids, 0.1)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)

        output = output * position_ids

        output = torch.sum(output) / (torch.sum(mask) + epsion)

        return output

'''
class LanguageCriterion(nn.Module):
    def __init__(self):
        super(LanguageCriterion, self).__init__()

        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').cuda()

    def forward(self, input, target):
        # input (batch, length, vocab)
        # target (batch, length)

        batch = input.size(0)
        length = input.size(1)

        output = self.crit(input.view(-1, input.size(-1)), target.view(-1))   # (batch, length)
        output = output.view(batch, length)

        position_ids = torch.arange(start=1, end=length+1, dtype=torch.float, device=input.device).flip(0)
        position_ids = torch.pow(position_ids, 0.1)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)

        loss = output * position_ids

        return output.mean()
'''

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def Frobenius(mat, mask):
    mask = mask.contiguous().view(-1)
    epsion = 1e-8
    size = mat.size()
    if len(size) == 3:  # batched matrix
        mat = mat.view(size[0], -1)
        ret = (torch.sum(mat ** 2, 1) + epsion) ** 0.5 * mask
        return torch.sum(ret) / (torch.sum(mask) + epsion)
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')


class KBCriterion(nn.Module):

    def __init__(self):
        super(KBCriterion, self).__init__()

    def forward(self, probs, co_matrix):
        batch_size = probs.size(0)
        num_classes = probs.size(1)

        diff_probs = probs.new_zeros(batch_size, num_classes, num_classes)
        for i in range(num_classes):
            diff_probs[:, i, :] = probs - probs[:, i].unsqueeze(1)

        diff_probs = torch.pow(diff_probs, 2)

        # loss = probs.new_zeros(batch_size, num_classes, num_classes)
        # for i in range(num_classes):
        #     for j in range(i, num_classes):
        #         loss[:, i, j] = 1.0 * torch.pow(probs[:, i] - probs[:, j], 2) * co_matrix[i][j]

        epsion = 1e-8
        mask = co_matrix > 0

        loss = diff_probs * co_matrix.unsqueeze(0)

        return torch.sum(torch.mean(loss, dim=0)) / (torch.sum(mask.float()) + epsion)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))