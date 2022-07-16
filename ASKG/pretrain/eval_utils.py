# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
import pickle
from json import encoder
import random
import string
import time
import os
import misc.utils as utils



import sys

sys.path.append('./pycocoevalcap')

from bleu.bleu import Bleu
from rouge.rouge import Rouge
from cider.cider import Cider

# from meteor.meteor import Meteor


def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores


def save_data(truth2id, sample2id, save_to, split):
    with open(save_to + 'truth2sample.' + split + '.txt', 'w') as f:
        for k, v in truth2id.items():
            f.writelines(k + '\n')
            f.writelines('Truth: ' + str(v) + '\n')
            f.writelines('Sample: ' + str(sample2id[k]) + '\n')


def evaluate(id2truth, id2sample, save_to='./results/', split='val'):
    # print (id2truth)
    # print (id2sample)
    # make dictionary
    print('Evaluating...')

    # compute bleu score
    final_scores = score(ref=id2truth, hypo=id2sample)

    # print the scores
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])
    print('Bleu_3:\t', final_scores['Bleu_3'])
    print('Bleu_4:\t', final_scores['Bleu_4'])
    # print('METEOR:\t', final_scores['METEOR'])
    print('ROUGE_L:', final_scores['ROUGE_L'])
    print('CIDEr:\t', final_scores['CIDEr'])

    cands = [''.join(v).replace(' ', '') for k, v in id2sample.items()]
    refs = [''.join(id2truth[k]).replace(' ', '') for k, v in id2sample.items()]
   
    from bert_score import score as BERTSCORE

    P, R, F1 = BERTSCORE(cands, refs, bert="bert-base-chinese")
    print('BERTSCORE:\t', P.mean(), R.mean(), F1.mean())

    # save data
    # save_data(truth2id=id2truth, sample2id=id2sample, save_to=save_to, split=split)

    return final_scores

def evaluate_en(id2truth, id2sample, save_to='./results/', split='val'):
    # print (id2truth)
    # print (id2sample)
    # make dictionary
    print('Evaluating...')

    # compute bleu score
    final_scores = score(ref=id2truth, hypo=id2sample)

    # print the scores
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])
    print('Bleu_3:\t', final_scores['Bleu_3'])
    print('Bleu_4:\t', final_scores['Bleu_4'])
    # print('METEOR:\t', final_scores['METEOR'])
    print('ROUGE_L:', final_scores['ROUGE_L'])
    print('CIDEr:\t', final_scores['CIDEr'])

    cands = [''.join(v).replace(' ', '') for k, v in id2sample.items()]
    refs = [''.join(id2truth[k]).replace(' ', '') for k, v in id2sample.items()]
   
    from bert_score import score as BERTSCORE

    P, R, F1 = BERTSCORE(cands, refs, bert="bert-base-uncased")
    print('BERTSCORE:\t', P.mean(), R.mean(), F1.mean())

    # save data
    # save_data(truth2id=id2truth, sample2id=id2sample, save_to=save_to, split=split)

    return final_scores
