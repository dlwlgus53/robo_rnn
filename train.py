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


def copy_target(target,length, train):
    np_target = target.cpu().numpy()
    for_expand = np.ones(length)
    long_target = torch.from_numpy(np_target * for_expand)
    long_target = long_target.view(np_target.shape[0],length,-1)
    long_target = long_target.type(torch.cuda.LongTensor)
    
    if train:
        return long_target.cuda()
    else:
        return long_target

def get_masked_loss(output, target, mask):
    losses = [[]*n for n in range(target.shape[0])]
    #loss_matrix = []
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            loss = F.cross_entropy(output[i][j].unsqueeze(0), target[i][j])
            losses[i].append(loss)
   
    
    # grad true until lossses


    masked_loss = [[]*n for n in range(target.shape[0])]
    
    # masked_loss = losses * mask
    
    for i in range (target.shape[0]):
        masked = torch.tensor(losses[i]).cuda() * torch.as_tensor(mask[i]).cuda()
        masked = torch.tensor(masked, requires_grad = True).cuda()
        #masked = masked.clone().detach().requires_grad_(True),
        masked_loss[i]=masked
    
    #print(masked_loss)
    return masked_loss

def get_losses_sum(losses):
    
    
  
    for i in range(len(losses)):
        losses_sum = torch.sum(losses[i])


    return losses_sum
def train_main(model, args, train_data, test_data, optimizer):
    patience = args.patience
    bad_counter = 0
    best_f1 = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, epoch, train_data, optimizer, args)
        print(epoch % args.val_freq)
        if epoch % args.val_freq == 0: # at each valid frequency
            acc, f1 = test(model, args, test_data)
            print('-' * 90)
            print("epoch {:d}".format(epoch))
            print(acc)
            print(f1)
            #print("epoch {:d} | acc {:.2f} | f1 {:.2f}".format(epoch, acc, f1))
            print('-' * 90)
            '''
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
            '''

def train(model, epoch, train_data, optimizer, args):
    total_batch_loss = 0
    log_interval = 300

    model.train()

    batch_size = args.batch_size 
    for batch_idx in range(train_data.n_batches):
        if batch_idx % log_interval == 0:
            print(batch_idx, train_data.n_batches)
        data, mask,target = train_data.getBatch(batch_idx)
        data, mask,target = data.to(args.device), mask.to(args.device), target.to(args.device)#.squeeze()??
        batch_size = data.shape[0]
        len = data.shape[1]
        
        data = data.view(batch_size, len, 1) #(Batchsize, maxdaylen, 1)
        optimizer.zero_grad()
        
        output = model(data)

        long_target = copy_target(target, len, train =1)
        #print(output)
        losses = get_masked_loss(output, long_target, mask)
        #target = target.view(1, batch_size).squeeze()
        #loss = F.cross_entropy(output,target)#TODO right??
       
         
        losses_sum = get_losses_sum(losses)

        losses_sum.backward()
        
        
        optimizer.step()
        '''
        loss_ += loss.item()
        train_loss += loss.item()
        '''  
        
        total_batch_loss += losses_sum/torch.sum(mask)
        '''
        backward() where..?
        '''



    return total_batch_loss / (batch_idx + 1)


def get_predictions(output, target):
    predictions = [[]*n for n in range(target.shape[0])]
  
 #loss_matrix = []
    '''
    prediction_ = torch.argmax(output, dim=1)
    prediction = torch.cat((prediction, prediction_), 0)

    targets = torch.cat((targets, target), 0)
    '''
    
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            prediction_ = torch.argmax(output[i][j].unsqueeze(0), dim = 1)
            if j ==0 :
                predictions[i] = prediction_.cpu()
            else : 
                predictions[i] = torch.cat((predictions[i], prediction_.cpu()),0)
            



    return predictions


def get_acc(targets, predictions, num):

    #print(len(targets), len(targets[0]), len(targets[0][0]))
    acc=[[] * n for n in range(len(targets)) ]
    sum =0 
    for i in range(len(targets)): # dataLen / batch size = batch index
        for j in range(len(targets[0])):# batch size data. Can get acc by day but not now
            acc_ = accuracy_score(targets[i][j].cpu(),predictions[i][j])
            if j == 0:
                acc[i] = acc_
            else:
                acc[i] =np.append(acc[i],acc_)
        sum += np.sum(acc[i])
    return sum/num

def get_f1(targets, predictions):
    f1=[[] * n for n in range(len(targets)) ]
    sum =0
    for i in range(len(targets)):
        for j in range(len(targets[0])):
            f1_ =  f1_score(targets[i][j].cpu(), predictions[i][j], average = "binary")
            if j ==0:
                f1[i] = f1_
            else:
                f1[i] = np.append(f1[i], f1_)
        sum += np.sum(f1[i])
    return sum
 
def test(model, args, test_data, valid=1):
    model.eval()
    correct = 0
    flag = 0
    total_num = test_data.n_num
    batch_size = args.batch_size
    predictions = [[]*n for n in range(test_data.n_batches)]
    targets = [[]*n for n in range(test_data.n_batches)]
    if valid:
        with torch.no_grad():
            for batch_idx in range(test_data.n_batches):
                data,mask, target = test_data.getBatch(batch_idx)
                len = data.shape[1]
                data,mask, target = data.to(args.device),mask.to(args.device), target.to(args.device).long()
                data = data.view(batch_size, len, 1)# (Batchsize, maxdaylen, 1)
                output = model(data)

                long_target_ = copy_target(target, len, train =0 )
                predictions_ = get_predictions(output, long_target_)
                
                # long_target_ = long_target_.cpu()
                # predctions_ = predictions_.cpu()
                '''
                if flag:
                    predictions = torch.cat((predictions, predictions_), 0)

                    targets = torch.cat((targets, long_target_), 0)

                else:
                    predictions = predictions_
                    targets = long_target_
                    flag = 1
                '''
                predictions[batch_idx] = predictions_
                targets[batch_idx] = long_target_ 
            '''
            targets = targets.cpu()
            prediction = prediction.cpu()
            

            acc = accuracy_score(targets, prediction)
            f1 = f1_score(targets, prediction, average="binary")
            '''

            acc = get_acc(targets, predictions, total_num)
            f1 = get_f1(targets, predictions)
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
                                       
