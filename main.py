from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import datetime
import pandas as pd
from torch.utils.data import DataLoader

#Import my model and Dataset
from dataload import BatchDataset
from model import RNN
from train import test, train, train_main
import sys
import os

parser = argparse.ArgumentParser(description='NI: Stas Action Classification')
parser.add_argument('--data', type=str, default="norm",
                    help='dataset for experiments varied by charactersitic of dummies (dummy, dummy0.2, dummy0.1, dummy2)')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--optim', type=str, default="RMSprop")
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate (no default)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int, default=3, metavar='P',
                    help='Early Stopping Patience')

parser.add_argument('--limit', type=int, default=10,
                    help='limit of looking day')


parser.add_argument('--val_freq', type=int, default=1, metavar='VF',
                    help='Validation Test Frequency')

parser.add_argument('--saveto', type=str, default='./saveto')

parser.add_argument('--VNF', type=str, default='firewall')

parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--expName', type=str, default="sample")
parser.add_argument('--input_dim', type=int, default=1) #maybe..?
parser.add_argument('--output_dim', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=5)
parser.add_argument('--layers', type=int, default=1)

args = parser.parse_args()
args.device = torch.device("cuda")
args.test_batch_size = args.batch_size

csv_path = os.path.join("../data", args.data, "sample.csv")
train_data = BatchDataset(csv_path, args.batch_size, args.limit)

csv_path = os.path.join("../data", args.data, "sample.csv")
valid_data = BatchDataset(csv_path, args.batch_size, args.limit)

csv_path = os.path.join("../data", args.data, "sample.csv")
test_data = BatchDataset(csv_path, args.batch_size, args.limit)


'''
batch data

X,Y = train_data.getBatch(i)
train-data.n_batches

'''


n_exp = 1
accs =[]
f1s = []

for n in range(n_exp):
    model = RNN(args)
    print("Setting Model Complete")
    model.to(args.device)

    optimizer = "optim." + args.optim
    optimizer = eval(optimizer)(model.parameters(), lr=args.lr)
    #optimizer = optim.RMSprop(model.parameters())
    #optimizer = optim.Adam(model.parameters())
    #oprimizer = optim.SGD(model.parameters())

    print("Setting optimizer Complete")

    train_main(model, args, train_data, valid_data, optimizer)
    acc, f1 = test(model, args, test_data)

    accs.append(acc)
    f1s.append(f1)

print("=" * 90)
print("results on testset at each trials:")

for i in range(n_exp):
    print("exp {:d} | acc {:.3f} | f1 {:.3f}".format(i+1, accs[i], f1s[i]))

total_acc = sum(accs) / n_exp
total_f1 = sum(f1s) / n_exp

print("-" * 90)
print("total scores")
print("acc {:.3f} | f1 {:.3f}".format(sum(accs)/n_exp, sum(f1s)/n_exp))
print("=" * 90)

# make logs of results


with open(os.path.join(args.saveto, args.expName + ".result"), 'a') as fp:
    #fp.write(str(optim) + ',' + str(args.lr) + ',' + hiddens + ',' + str(args.patience) + ',' + str(args.batch_size) + ','+ str(total_acc) ',' + str(total_f1))
    fp.write("{:s},{:f},{:d},{:d},{:f},{:f}\n".format(args.optim, args.lr, args.patience, args.batch_size, total_acc, total_f1))

