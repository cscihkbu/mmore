import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layers

def multi_class_cross_entropy_loss(predictions, labels):
    loss = -torch.mean(torch.sum(torch.sum(labels * torch.log(predictions), dim=1), dim=1))
    return loss

class MMORE_DXRX(nn.Module):
    def __init__(self, args):
        super(MMORE_DXRX, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.drugontoEmb = layers.OntoEmb(self.drugVocabSize, self.drugnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize+self.drugVocabSize)
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.EHRdrugEmb = nn.Embedding(self.drugVocabSize+1, self.EHREmbDim, padding_idx=self.drugVocabSize)
        self.attn = nn.Linear(2*self.ontoEmbDim+2*self.EHREmbDim, self.ptattnDim)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)

    def forward(self, inputs):
        (dxseqs, drugseqs, dx_onehot, drug_onehot, 
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList) = inputs
        dxontoInputs = (dxLeavesList, dxAncestorsList)
        drugontoInputs = (drugLeavesList, drugAncestorsList)
        dxseqs = dxseqs.to(self.device)
        drugseqs = drugseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        drug_onehot = drug_onehot.to(self.device)
        dx_num = list(dxseqs.size())[2]
        drug_num = list(drugseqs.size())[2]
        dxEHREmb = self.EHRdxEmb(dxseqs)
        drugEHREmb = self.EHRdrugEmb(drugseqs)
        EHREmb = torch.cat((dxEHREmb, drugEHREmb), dim=2)
        dxALLontoEmb = self.dxontoEmb(dxontoInputs)
        drugALLontoEmb = self.drugontoEmb(drugontoInputs)
        dxOntoEmb = dxALLontoEmb[dxseqs]
        drugOntoEmb = drugALLontoEmb[drugseqs]
        ontoEmb = torch.cat((dxOntoEmb, drugOntoEmb), dim=2)
        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        drugEHRVEmb= F.normalize(torch.sum(self.EHRdrugEmb(drugseqs), dim=2), p=2, dim=2)
        EHRVEmb = dxEHRVEmb+drugEHRVEmb
        dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
        drugontoVEmb = torch.matmul(drug_onehot.permute(1,0,2), drugALLontoEmb[:-1])
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = torch.cat((dx_onehot.permute(1,0,2), drug_onehot.permute(1,0,2)), dim=2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        ontoVEmb = F.normalize(dxontoVEmb, p=2, dim=2)+F.normalize(drugontoVEmb, p=2, dim=2)
        vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        dxdrugEmb = torch.cat((EHREmb,ontoEmb), dim=3)
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num+drug_num,1)
        attnInput = torch.cat((vs_emb, dxdrugEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxdrugEmb), 2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2)
        return DP_result, cooccur_loss*10
    
class MMORE_DX(nn.Module):
    def __init__(self, args):
        super(MMORE_DX, self).__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)
        self.dxontoEmb = layers.OntoEmb(self.dxVocabSize, self.dxnumAncestors, self.ontoEmbDim, self.ontoattnDim)        
        self.EHRdxEmb = nn.Embedding(self.dxVocabSize+1, self.EHREmbDim, padding_idx=self.dxVocabSize)
        self.attn = nn.Linear(2*self.EHREmbDim+2*self.ontoEmbDim, self.ptattnDim)
        self.attnCombine = nn.Linear(self.ptattnDim, 1)
        self.cooccurLinear = nn.Linear(self.EHREmbDim, self.dxVocabSize)
        self.dpPredLinear = nn.Linear(self.ontoEmbDim+self.EHREmbDim, self.dpLabelSize)

    def forward(self, inputs):
        (dxseqs, dx_onehot, dxLeavesList, dxAncestorsList,) = inputs
        dxontoInputs = (dxLeavesList, dxAncestorsList)
        dxseqs = dxseqs.to(self.device)
        dx_onehot = dx_onehot.to(self.device)
        dx_num = list(dxseqs.size())[2]
        dxALLontoEmb = self.dxontoEmb(dxontoInputs)
        dxOntoEmb = dxALLontoEmb[dxseqs]
        dxontoVEmb = torch.matmul(dx_onehot.permute(1,0,2), dxALLontoEmb[:-1])
        ontoVEmb = dxontoVEmb
        dxEHREmb = self.EHRdxEmb(dxseqs)
        dxEHRVEmb = F.normalize(torch.sum(self.EHRdxEmb(dxseqs), dim=2), p=2, dim=2)
        EHRVEmb = dxEHRVEmb
        cooccurU = F.softmax(self.cooccurLinear(EHRVEmb), dim=2).contiguous()
        vonehot = dx_onehot.permute(1,0,2).contiguous()
        cooccur_loss = multi_class_cross_entropy_loss(cooccurU, vonehot)
        vs_emb = torch.cat((ontoVEmb, EHRVEmb), dim=2)
        dxEmb = torch.cat((dxEHREmb, dxOntoEmb), dim=3)
        vs_emb = torch.unsqueeze(vs_emb, dim=2).repeat(1,1,dx_num,1)
        attnInput = torch.cat((vs_emb, dxEmb), dim=3)
        mlpOutput = torch.tanh(self.attn(attnInput))
        preAttention = self.attnCombine(mlpOutput)
        attention = F.softmax(preAttention, dim=2)
        vs_emb = torch.sum(torch.mul(attention.repeat(1,1,1,self.EHREmbDim+self.ontoEmbDim), dxEmb), dim=2)
        vs_emb_dp = F.normalize(vs_emb, p=2, dim=2)
        DP_result = F.softmax(self.dpPredLinear(torch.tanh(vs_emb_dp)), dim=2) 
        return DP_result, cooccur_loss*10
