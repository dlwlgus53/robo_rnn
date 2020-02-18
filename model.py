# MNSIT code for Dropout paper

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(args.input_dim,args.hidden_dim, args.layers, batch_first=True) 
        self.fc = torch.nn.Linear(args.hidden_dim, args.output_dim, bias=True)
        self.batch_size = args.batch_size
    def forward(self, x):
        try : 
            x, _status = self.rnn(x)
        except :
            import pdb; pdb.set_trace()
        after_fc = [[] for n in range(x.shape[0])]
        for i in range(x.shape[0]):#batch size
            for j in range(x.shape[1]):#day limit length
                after_fc[i].append(self.fc(x[i,j,:]))
                '''
                if j == 0:
                    after_fc[i] = self.fc(x[i,j,:])
                else : 
                    after_fc[i] = torch.cat((after_fc[i],self.fc(x[i,j,:])))
                '''
            
        after_softmax = [[] for n in range(x.shape[0])]
        for i in range(x.shape[0]):#batch size
            for j in range(x.shape[1]):# day limit length
                temp = F.softmax(after_fc[i][j], dim=0)
                after_softmax[i].append(temp)
                 
                #print(F.softmax(after_fc[i][j], dim=0))
                '''
                if j==0:
                    after_softmax[i] = (F.softmax(after_fc[i][j],dim =0)).unsqueeze(0)
                else :
                    print(after_softmax[i])
                    print(F.softmax(after_fc[i][j],dim=0))
                    print(after_softmax[i].size())
                    after_softmax[i] = torch.cat((after_softmax[i], F.softmax(after_fc[i][j],dim=0)))
                '''


        #print(after_softmax)
        return after_softmax


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()

        self.FF = nn.Sequential(
                    nn.Linear(args.horizon, args.hidden1), nn.Sigmoid(),
                    nn.Linear(args.hidden1, args.hidden2), nn.Sigmoid(),
                    nn.Linear(args.hidden2, args.hidden3), nn.Sigmoid(),
                    nn.Linear(args.hidden3, 2)
                    )

    def forward(self, x):
        out = self.FF(x)

        '''
        print('-' * 90)
        print(out)
        print("shape after fc: ", out.shape)
        print('-' * 90)
        '''

        out = F.softmax(out, dim=1)

        '''
        print('-' * 90)
        print(out)
        print("shape after softmax: ", out.shape)
        print('-' * 90)
        sys.exit(0)
        '''
        print(out)

        return out
