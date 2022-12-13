from cProfile import label
#import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import os 


class MyDataset(Dataset):

    def __init__(self, filename):
        super().__init__()
        
        #Data is read from the text file X data and F(X) is stored in data and labels tensors 
        if1=os.path.join(filename)  
        dataL=[] ; labelL=[]
        with open(if1, encoding='utf-8', mode ='r') as ifile1:
            reader =csv.reader(ifile1, delimiter ='\t')
            for line in reader:
                dataL.append([float(line[0]),float(line[1]),float(line[2]),float(line[3])])
                labelL.append([float(line[4])])
                
        # convert list to tensor        
        self.data=torch.Tensor(dataL)
        self.labels=torch.Tensor(labelL)
        
        return
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
   
    
if __name__ == '__main__':
    my_dataset = MyDataset(99,99,99)
    