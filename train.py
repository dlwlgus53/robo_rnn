# MNSIT code for Dropout paper
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import csv
import time
import sys

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as f1_metrics

'''
def masking(data, horizon = 49):

    mask_rows = []


    for n in range(data.shape[0]):
        mask_row = []
        throw = 0
        for i in range(horizon):
            # first entry is nan
            if np.isnan(data[n,i]) and i==0:
                throw = 1
                break

            # other entries are nan
            if np.isnan(data[n,i]):
                mask_row.append(0.)
            else:
                # normal entry
                mask_row.append(1)

        if throw == 0:
            mask_rows.append(mask_row)


    x = np.vstack(mask_rows)
    return x
'''


def train_main(model, args, train_data, test_data, optimizer):
    patience = args.patience
    bad_counter = 0
    best_f1 = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, epoch, train_data, optimizer, args)

        if epoch % args.val_freq == 0: # at each valid frequency
            acc, f1 = test(model, args, test_data)
            print('-' * 90)
            print("epoch {:d} | acc {:.2f} | f1 {:.2f}".format(epoch, acc, f1))
            print('-' * 90)

            if f1 > best_f1:
                best_f1 = f1
                print("Found Best Model")
                save_checkpoint(model, True, args)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter > patience :
                # e-stop
                print('Early Stopping...')
                break


def train(model, epoch, train_data, optimizer, args):
    loss_ = 0
    train_loss = 0
    log_interval = 300

    model.train()

    batch_size = args.batch_size 
    for batch_idx in range(train_data.n_batches):
        data, target = train_data.getBatch(batch_idx)
        data, target = data.to(args.device), target.to(args.device)#.squeeze()??
        batch_size = data.shape[0]
        len = data.shape[1]
        
        data = data.view(batch_size, len, 1) #(Batchsize, maxdaylen, 1)
        optimizer.zero_grad()
        
        output = model(data)
        target = target.view(1, batch_size).squeeze()
        loss = F.cross_entropy(output,target)#TODO right??
        loss.backward()
        optimizer.step()
        loss_ += loss.item()
        train_loss += loss.item()
        
        '''
        loss_ = criterion(output_flat, y.view(-1))
        loss = torch.sum(loss*mask)
        loss.backward()
        total_loss = torch.sum(loss * mask)/torch.sum(mask)
        
        backward() where..?
        '''
        if batch_idx % log_interval == (log_interval-1):
            print("batch_idx {:d} | loss {:.6f}".format(batch_idx+1, loss_ / log_interval))
            loss_ = 0



    return train_loss / (batch_idx + 1)

def test(model, args, test_data, valid=1):#TODO valid 0??
    model.eval()
    correct = 0
    flag = 0
    batch_size = args.batch_size
    if valid:
        with torch.no_grad():
            for batch_idx in range(test_data.n_batches):
                data, target = test_data.getBatch(batch_idx)
                len = data.shape[1]
                data, target = data.to(args.device), target.to(args.device).long()
                data = data.view(batch_size, len, 1)# (Batchsize, maxdaylen, 1)
                output = model(data)

                if flag:
                    prediction_ = torch.argmax(output, dim=1)
                    prediction = torch.cat((prediction, prediction_), 0)

                    targets = torch.cat((targets, target), 0)

                else:
                    prediction = torch.argmax(output, dim=1)
                    targets = target
                    flag = 1

            targets = targets.cpu()
            prediction = prediction.cpu()
            acc = accuracy_score(targets, prediction)
            f1 = f1_score(targets, prediction, average="binary")

        return acc, f1

def save_checkpoint(model, is_best, args):
    if not os.path.exists(args.saveto):
         print('Make Results Directory (Path: {})'.format(args.saveto))
         os.makedirs(args.saveto)

    filename = str(args.expName) + '.pth'

    path = os.path.join(args.saveto, filename)
    torch.save(model, path)
    if is_best:
        shutil.copyfile(path, path + '.best')

if __name__ == '__main__':
    pass
                                       
