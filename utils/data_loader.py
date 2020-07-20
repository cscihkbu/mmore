import numpy as np
import random
import math
import pickle
import torch
from torch.autograd import Variable

_TEST_RATIO = 0.1
_TRAIN_RATIO = 0.8
_VALIDATION_RATIO = 0.1


def dx(dxSeqFile, dpLabelFile):
    dxSeqs = np.array(pickle.load(open(dxSeqFile, 'rb')))
    dpLabels = np.array(pickle.load(open(dpLabelFile, 'rb')))

    np.random.seed(0)
    dataSize = len(dxSeqs)
    ################## Random ##################
    ind = np.random.permutation(dataSize)
    ################## END ##################
    # ind = np.arange(dataSize)
    # print('ind1:', ind.shape, ind)

    # train: 0~80%, valid: 80~90%, test: 90~100%
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)
    nTrain = int((1-_TEST_RATIO-_VALIDATION_RATIO)*dataSize)
    # nTrain = int(_TRAIN_RATIO * dataSize)

    test_indices = ind[(dataSize-nTest):]
    valid_indices = ind[(dataSize-nTest-nValid):(dataSize-nTest)]
    train_indices = ind[:(dataSize-nTest-nValid)]

    train_set_x = dxSeqs[train_indices]
    train_set_dp_y = dpLabels[train_indices]
    test_set_x = dxSeqs[test_indices]
    test_set_dp_y = dpLabels[test_indices]
    valid_set_x = dxSeqs[valid_indices]
    valid_set_dp_y = dpLabels[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_dp_y = [train_set_dp_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_dp_y = [valid_set_dp_y[i] for i in valid_sorted_index] 

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_dp_y = [test_set_dp_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_dp_y)
    valid_set = (valid_set_x, valid_set_dp_y)
    test_set = (test_set_x, test_set_dp_y)

    return train_set, valid_set, test_set

def dxrx(dxSeqsFile, drugSeqsFile, dpLabelFile):
    drugSeqs = np.array(pickle.load(open(drugSeqsFile, 'rb')))
    dxSeqs = np.array(pickle.load(open(dxSeqsFile, 'rb')))
    dpLabels = np.array(pickle.load(open(dpLabelFile, 'rb')))

    np.random.seed(0)
    dataSize = len(dxSeqs)
    ################## Random ##################
    ind = np.random.permutation(dataSize)
    ################## END ##################
    # ind = np.arange(dataSize)
    # print('ind1:', ind.shape, ind)

    # train: 0~80%, valid: 80~90%, test: 90~100%
    nTest = int(_TEST_RATIO * dataSize)
    nValid = int(_VALIDATION_RATIO * dataSize)
    nTrain = int((1-_TEST_RATIO-_VALIDATION_RATIO)*dataSize)
    # nTrain = int(_TRAIN_RATIO * dataSize)

    test_indices = ind[(dataSize-nTest):]
    valid_indices = ind[(dataSize-nTest-nValid):(dataSize-nTest)]
    train_indices = ind[:(dataSize-nTest-nValid)]

    train_set_dx_x = dxSeqs[train_indices]
    train_set_drug_x = drugSeqs[train_indices]
    train_set_dp_y = dpLabels[train_indices]
    test_set_dx_x = dxSeqs[test_indices]
    test_set_drug_x = drugSeqs[test_indices]
    test_set_dp_y = dpLabels[test_indices]
    valid_set_dx_x = dxSeqs[valid_indices]
    valid_set_drug_x = drugSeqs[valid_indices]
    valid_set_dp_y = dpLabels[valid_indices]

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_dx_x)
    train_set_dx_x = [train_set_dx_x[i] for i in train_sorted_index]
    train_set_drug_x = [train_set_drug_x[i] for i in train_sorted_index]
    train_set_dp_y = [train_set_dp_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_dx_x)
    valid_set_dx_x = [valid_set_dx_x[i] for i in valid_sorted_index]
    valid_set_drug_x = [valid_set_drug_x[i] for i in valid_sorted_index]
    valid_set_dp_y = [valid_set_dp_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_dx_x)
    test_set_dx_x = [test_set_dx_x[i] for i in test_sorted_index]
    test_set_drug_x = [test_set_drug_x[i] for i in test_sorted_index]
    test_set_dp_y = [test_set_dp_y[i] for i in test_sorted_index]

    train_set = (train_set_dx_x, train_set_drug_x, train_set_dp_y)
    valid_set = (valid_set_dx_x, valid_set_drug_x, valid_set_dp_y)
    test_set = (test_set_dx_x, test_set_drug_x, test_set_dp_y)

    return train_set, valid_set, test_set
