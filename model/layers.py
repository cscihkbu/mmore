"""
Additional layers.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')

class OntoEmb(nn.Module):
    """docstring for OntoEmb"""
    def __init__(self, dxVocabSize, dxnumAncestors, dxEmbDim, attnDim):
        super(OntoEmb, self).__init__()
        self.dxVocabSize = dxVocabSize
        self.dxnumAncestors = dxnumAncestors
        self.dxEmbDim = dxEmbDim
        self.attnDim = attnDim
        self.dxEmb = nn.Embedding(self.dxVocabSize+self.dxnumAncestors, self.dxEmbDim)
        self.attn = nn.Linear(2*self.dxEmbDim, self.attnDim)
        self.attnCombine = nn.Linear(self.attnDim, 1)

    def forward(self, ontoInput):
        leavesList, ancestorsList = ontoInput
        tempAllEmb = []
        for leaves, ancestors in zip(leavesList, ancestorsList):
            leavesEmb = self.dxEmb(leaves.to(device))
            ancestorsEmb = self.dxEmb(ancestors.to(device))
            attnInput = torch.cat((leavesEmb, ancestorsEmb), dim=2)
            mlpOutput = torch.tanh(self.attn(attnInput))
            preAttn = self.attnCombine(mlpOutput)
            attn = F.softmax(preAttn, dim=1)
            tempEmb = torch.sum(ancestorsEmb*attn, dim=1)
            tempAllEmb.append(tempEmb)
        tempAllEmb.append(torch.zeros(1, self.dxEmbDim).to(device))
        allEmb = torch.cat([i for i in tempAllEmb], dim=0)
        return allEmb

