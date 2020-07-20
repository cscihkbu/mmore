import argparse, random, time, pickle, math
import numpy as np
import torch
import torch.nn.functional as F
from operator import mul

from model import models
from utils import data_loader
from utils import data_helper

def calculate_vocabSize(file):
    sequences = pickle.load(open(file, 'rb'))
    codeDict = {}
    for patient_title in sequences:
        for visit_word in patient_title:
            for icd_char in visit_word:
                codeDict[icd_char] = ''
    return len(codeDict)

def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = list(tree.values())[0][1]
    return rootCode

def build_tree(treeFile):
    treeMap = pickle.load(open(treeFile, 'rb'))
    ancestors = np.array(list(treeMap.values())) 
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves)
    leaves = torch.LongTensor(leaves)
    ancestors = torch.LongTensor(ancestors)
    return leaves, ancestors

parser = argparse.ArgumentParser()
parser.add_argument('--dxSeqsFile', type=str, default='./inputs/dx.seqs')
parser.add_argument('--drugSeqsFile', type=str, default='./inputs/rx.seqs')
parser.add_argument('--dxtreeFile', type=str, default='./inputs/dx')
parser.add_argument('--drugtreeFile', type=str, default='./inputs/rx')
parser.add_argument('--dpLabelFile', type=str, default='./inputs/dp.labels')
parser.add_argument('--EHREmbDim', type=int, default=400)
parser.add_argument('--ontoEmbDim', type=int, default=400)
parser.add_argument('--ontoattnDim', type=int, default=100)
parser.add_argument('--ptattnDim', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--topk', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--LR', type=float, default=1.0)
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--save', type=str, default='./outputs/model.pt')
args = parser.parse_args()
args.dxVocabSize = calculate_vocabSize(args.dxSeqsFile)
args.drugVocabSize = calculate_vocabSize(args.drugSeqsFile)
args.dpLabelSize = calculate_vocabSize(args.dpLabelFile)
args.dxnumAncestors = get_rootCode(args.dxtreeFile+'.level2.pk')-args.dxVocabSize+1
args.drugnumAncestors = get_rootCode(args.drugtreeFile+'.level3.pk')-args.drugVocabSize+1
args.allDxs = [dx for dx in range(args.dxVocabSize)][1:]
args.use_cuda = torch.cuda.is_available() and args.use_gpu
args.device = torch.device('cuda' if args.use_cuda else 'cpu')
torch.cuda.manual_seed_all(args.seed)

# ##############################################################################
# Load data
################################################################################

train_set, valid_set, test_set = data_loader.dxrx(args.dxSeqsFile, args.drugSeqsFile, args.dpLabelFile)

dxLeavesList = []
dxAncestorsList = []
drugLeavesList = []
drugAncestorsList = []

for i in range(5, 1, -1): # An ICD9 diagnosis code can have at most five ancestors (including the artificial root) when using CCS multi-level grouper. 
    leaves, ancestors = build_tree(args.dxtreeFile+'.level'+str(i)+'.pk')
    dxLeavesList.append(leaves)
    dxAncestorsList.append(ancestors)
for i in range(5, 0, -4): 
    leaves, ancestors = build_tree(args.drugtreeFile+'.level'+str(i)+'.pk')
    drugLeavesList.append(leaves)
    drugAncestorsList.append(ancestors)

# ##############################################################################
# Build model
# ##############################################################################

mmore_model = models.MMORE_DXRX(args)
print('mmore_model:', mmore_model)

optimizer = torch.optim.Adadelta([
    {'params': mmore_model.parameters()}
    ], 
    lr=args.LR, rho=0.95, weight_decay=0)

def get_criterion():
    return torch.nn.BCELoss(reduction='sum')

crit = get_criterion()

if args.use_cuda:
    mmore_model = mmore_model.to(args.device)
    crit = crit.to(args.device)

# ##############################################################################
# Training
# ##############################################################################

train_loss = []
valid_loss = []
test_loss = []

def get_dp_acc_train(args, crit, preds, targets):
    loss = crit(preds, targets)
    return loss

def get_dp_acc(args, crit, preds, targets):
    loss = crit(preds, targets)
    correct_dx_num = total_dx_num = 0
    patient_num = preds.size()[0]
    visit_num = preds.size()[1]
    dpLabelSize = preds.size()[2]
    preds = preds.view(patient_num*visit_num, -1)
    targets = targets.view(patient_num*visit_num, -1)
    pred_topk, pred_idx = torch.topk(preds, k=args.topk, dim=1)
    for v_pred_idx, v_tgt in zip(pred_idx, targets):
        v_tgts_idx = torch.nonzero(v_tgt)
        if list(v_tgts_idx.size()):
                total_dx_num += list(v_tgts_idx.size())[0]
        for idx in v_pred_idx:
            if idx in v_tgts_idx:
                correct_dx_num += 1
    return loss, correct_dx_num, total_dx_num

def evaluate(args, dataSet):
    mmore_model.eval()
    total_loss = patient_num = total_dxnum = correct_dxnum = 0
    batch_num = int(np.ceil(float(len(dataSet[0])) / float(args.batchSize))) - 1
    for bidx in random.sample(range(batch_num), batch_num):
        patient_num += args.batchSize
        dxseqs = dataSet[0][bidx*args.batchSize:(bidx+1)*args.batchSize]
        drugseqs = dataSet[1][bidx*args.batchSize:(bidx+1)*args.batchSize]
        dplabels = dataSet[2][bidx*args.batchSize:(bidx+1)*args.batchSize]
        dxseqs, dx_onehot = data_helper.get_seqs(dxseqs, args, codetype='dx')
        drugseqs, drug_onehot = data_helper.get_seqs(drugseqs, args, codetype='drug')
        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList)
        dp_result, cooccur_loss = mmore_model(inputs)
        labels_dp, dp_mask = data_helper.get_dp_mask(dplabels, args.dpLabelSize)
        pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
        pred_loss, batch_correct_dxnum, batch_total_dxnum = get_dp_acc(args, crit, pred_dp, labels_dp.to(args.device))
        batch_loss = pred_loss.add(cooccur_loss)
        total_loss += batch_loss.item()
        total_dxnum += batch_total_dxnum
        correct_dxnum += batch_correct_dxnum
    return total_loss/patient_num, correct_dxnum, total_dxnum, correct_dxnum/total_dxnum
    
def train(args, dataSet):
    mmore_model.train()
    total_loss = patient_num = 0
    batch_num = int(np.ceil(float(len(dataSet[0])) / float(args.batchSize))) - 1
    for bidx in random.sample(range(batch_num), batch_num):
        patient_num += args.batchSize  
        dxseqs = dataSet[0][bidx*args.batchSize:(bidx+1)*args.batchSize]
        drugseqs = dataSet[1][bidx*args.batchSize:(bidx+1)*args.batchSize]
        dplabels = dataSet[2][bidx*args.batchSize:(bidx+1)*args.batchSize] 
        dxseqs, dx_onehot = data_helper.get_seqs(dxseqs, args, codetype='dx')
        drugseqs, drug_onehot = data_helper.get_seqs(drugseqs, args, codetype='drug')
        inputs = (dxseqs, drugseqs, dx_onehot, drug_onehot,
            dxLeavesList, dxAncestorsList, drugLeavesList, drugAncestorsList)
        dp_result, cooccur_loss = mmore_model(inputs)
        labels_dp, dp_mask = data_helper.get_dp_mask(dplabels, args.dpLabelSize)
        pred_dp = torch.mul(dp_result, dp_mask.to(args.device))
        pred_loss = get_dp_acc_train(args, crit, pred_dp, labels_dp.to(args.device))
        batch_loss = pred_loss.add(cooccur_loss)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss/patient_num

# ##############################################################################
# Save Model
# ##############################################################################
best_valid_loss = None
best_test_acc = None
best_valid_acc = None
total_start_time = time.time()
try:
    print('-' * 70)
    for epoch in range(1, args.epochs+1):
        # Train
        epoch_start_time = time.time()
        trainLoss = train(args=args, dataSet=train_set)
        train_loss.append(trainLoss)
        print('| epoch: {:3d} (train) | loss: {:.2f} | time: {:2.0f}s'.format(epoch, trainLoss, time.time() - epoch_start_time))
        print('-' * 70)
        if epoch%2 == 0:
            # Validation
            validLoss, correct_dx, total_dx, validdpacc = evaluate(args=args, dataSet=valid_set)
            valid_loss.append(validLoss)
            print('| epoch: {:3d} (valid) | loss: {:.2f} | DPACC: {:.3f}% ({}/{})'.format(epoch, validLoss, validdpacc*100, correct_dx, total_dx))
            print('-' * 70)
            # Test
            testLoss, correct_dx, total_dx, testdpacc  = evaluate(args=args, dataSet=test_set)
            test_loss.append(testLoss)
            print('| epoch: {:3d} (test)  | loss: {:.2f} | DPACC: {:.3f}% ({}/{})'.format(epoch, testLoss, testdpacc*100, correct_dx, total_dx))
            print('-' * 70)
            if not best_valid_acc or not best_valid_acc > validdpacc:
                best_epoch_num = epoch
                best_valid_acc = validdpacc
                best_test_acc = testdpacc
                model_state_dict = mmore_model.state_dict()
                model_source = {
                    "settings": args,
                    "model": model_state_dict,
                }
                torch.save(model_source, args.save)
except KeyboardInterrupt:
    print("-"*70)
    print("Exiting from training early | cost time: {:5.2f} min".format((time.time() - total_start_time)/60.0))

print('Best epoch: {:3d} | DPACC: {:.5f} '.format(best_epoch_num, best_test_acc))
