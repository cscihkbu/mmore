import random
import numpy as np
import torch

from utils import const

def get_dp_mask(labels, labelSize):
    max_visit_num = np.max(np.array([len(p) for p in labels]))
    new_labels = []
    mask = []
    for p in labels:
        mask_p = []
        label_p = []
        for visit in p:
            mask_p.append(np.array([1]*labelSize))
            new_v = np.array([0]*labelSize)
            for label in visit:
                new_v[label] = 1
            label_p.append(new_v)
        if len(mask_p) < max_visit_num:
            mask_p.extend([np.array([0]*labelSize)] * (max_visit_num-len(label_p)))
            label_p.extend([np.array([0]*labelSize)] * (max_visit_num-len(label_p)))
        mask.append(np.array(mask_p[1:]))
        new_labels.append(np.array(label_p[1:]))
    return torch.FloatTensor(new_labels), torch.FloatTensor(mask)

def get_seqs(seqs, args, codetype):
    if codetype == 'dx':
        padid = const.PAD_DXID
        vocabSize = args.dxVocabSize
    elif codetype == 'drug':
        padid = const.PAD_DRUGID
        vocabSize = args.drugVocabSize
    else:
        padid = const.PAD_ID
    visit_num = np.array([len(p) for p in seqs])
    max_visit_num = np.max(visit_num)     
    code_num = []
    for p in seqs:
        max_dx_num = np.max(np.array([len(v) for v in p]))
        code_num.append(max_dx_num)
    max_code_num = np.max(np.array(code_num))
    new_seqs = []
    for p in seqs:
        new_p = []
        for v in p:
            new_v = v[:]
            if len(v) < max_code_num: 
                new_v.extend([padid]*(max_code_num-len(v)))
            new_p.append(new_v)
        if len(p) < max_visit_num:
            new_p.extend([[padid]*max_code_num]*(max_visit_num-len(p)))
        if max_visit_num > 1:
            new_seqs.append(new_p[:-1])
    lengths = np.array([len(seq) for seq in seqs]) - 1
    max_visit_num = np.max(lengths)
    if max_visit_num != 0:
        onehot = np.zeros((max_visit_num, args.batchSize, vocabSize))
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(onehot[:,idx,:], seq[:-1]): 
                xvec[subseq] = 1.
    else:
        new_seqs.append(new_p)
        onehot = np.zeros((1, args.batchSize, vocabSize))
        for idx, seq in enumerate(seqs):
            for xvec, subseq in zip(onehot[:,idx,:], seq): 
                xvec[subseq] = 1.
    return torch.LongTensor(new_seqs), torch.FloatTensor(onehot)

    