import os
import random
import numpy as np
import argparse
import pandas as pd
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

def trim_data(data, horizon=49):
    '''
    args
    data : npndarray, input and target combined
    horizon : int

    1) discard rows w/ zero length
    2) add zero padding for rows w/ length less
    3) trim parts of rows that exceed sequence length

    out : trimmed input, and target data
    '''

    trimmed_rows = []
    targets = []
    
    for n in range(data.shape[0]):
        trimmed_row = []
        target = 0 # temporary init
        throw = 0
        for i in range(horizon):
            # first entry is nan
            if np.isnan(data[n,i]) and i==0:
                throw = 1
                break

            # other entries are nan
            if np.isnan(data[n,i]):
                trimmed_row.append(0.)
            else:
                # normal entry
                trimmed_row.append(data[n,i].item())

        if throw == 0:
            trimmed_rows.append(trimmed_row)
            targets.append(data[n,-1].item())
           

    x = np.vstack(trimmed_rows)
    y = np.vstack(targets)
    #print("hello")
    #print(y)
    return x, y

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




class StasDataset(Dataset):
    def __init__(self, csv_path, horizon):
        df_data = pd.read_csv(csv_path, index_col = 0, header = 0) # dummy don't need options. norm need options
        data = np.asarray(df_data)
       
        self.row = data # save the row


        self.horizon = horizon
        x, y = trim_data(data, horizon)

        self.y = y.astype('int32').reshape(-1, 1)
        self.x = x.astype('float32')
        print(csv_path)
        print("input data matrix shape", self.x.shape)
        print("target data shape", self.y.shape)

    def __getitem__(self, index):
            return torch.from_numpy(self.x[index]).type(torch.float32),\
                    torch.from_numpy(self.y[index]).type(torch.long)

    def __len__(self):
            return self.x.shape[0]

    def getInputSize(self):
            return self.horizon
    
    def getMask(self):
            self.maskX = masking(self.row)
            return torch.from_numpy(self.maskX).type(torch.float32)



class BatchDataset():
    def __init__(self, csv_path, batch_size):
        self.batch_size = batch_size
        self.df_pd = pd.read_csv(csv_path, index_col =0 , header =0)
        self.df_np = self.remove_wrong_data(np.asarray(self.df_pd))
        np.random.shuffle(self.df_np)    #shuffle
        #print(self.df_np)
        
        self.datalen = self.df_np.shape[0]
        self.n_batches = int(self.datalen / self.batch_size)
        
        self.report()

        #self.x, self.y = self.cut_to_batch() 

        #remove wrong data from raw data
    def remove_wrong_data(self,data):
        cleans = []
        for n in range(data.shape[0]):
            if np.isnan(data[n,0]):
                continue
            cleans.append(data[n,:])
        return np.vstack(cleans)
        

        # cut to batach and devide X and Y
    def getBatch(self,i):
            
        batchedX = []
        batchedY = []
        start = i*self.batch_size
        end = (i+1)*self.batch_size
        batchedX = self.padding(self.df_np[start:end,:-1])
        batchedY = self.df_np[start:end,-1]
        return torch.from_numpy(np.vstack(batchedX)).type(torch.float32), torch.from_numpy(np.vstack(batchedY)).type(torch.long)
            
    def padding(self,data):
        maxLen = self.getMaxLen(data)
        rows = []
        for n in range(data.shape[0]):
            row = []
            for i in range(maxLen):
                if np.isnan(data[n,i]):
                    row.append(0)
                else:
                    row.append(data[n,i])
            row_np = np.array(row)
            if (i == 0):
                rows=row
            else:
                rows.append(row)
        return np.vstack(rows)
                                                                                                                                                                      

    def getMaxLen(self,data):
        max =0
        for i in range(data.shape[0]):
            nonzero=np.count_nonzero(~np.isnan(data[i,:]))
            if(max<nonzero):
                max = nonzero
        return max
            
            
    def report(self):
        print("shape of data")
        print(self.df_np.shape)
        print("number of batches")
        print(self.n_batches)


if __name__ == "__main__":
    data_path = "../data/norm"
    csv_path = os.path.join(data_path, "sample.csv")
    
    sample = BatchData(csv_path, batch_size =5)
    X,Y = sample.getBatch(3)
    print(X)
    print(Y)
