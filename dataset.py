# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:59:34 2020

@author: Rashid Haffadi
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from text import create_dataframe

class SequenceDataset(Dataset):

    def __init__(self, labels=None, df=None, from_pickle=False, path=None, max_words=50000, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        
        if from_pickle and path is not None:
            self.df = self._from_pickle(path)
        
        if df is None:
            if labels is not None:
                self.df = create_dataframe(labels, max_words=max_words)
        else:
            self.df = df
        self.x =     
    def _from_pickle(self, path):
        return pd.read_pickle(path)

    def __len__(self):
        return len(self.df)
    
    def _transform(self, xs):
        return np.vectorize(self.transform)(xs)

    def __getitem__(self, idx):
        
        if isinstance(idx, int):
            el = self.df.iloc[idx, :].tolist()
            if self.transform is not None:
                el[0] = self.transform(el[0])
    
            return el
        else:
            idx = list(idx)
            el = self.df.iloc[idx, :]
            el = np.array(el)
            if self.transform is not None:
                el[:, 0] = self._transform(el[:, 0])
            
            x = el[:, 0]
            y = el[:, 1]
            return (x, y.tolist())
    
    
if __name__ == "__main__":
    
    dataset = SequenceDataset(from_pickle=True, path="F:/workspace/PFE/Datasets/europarl/data.pkl")
    for i in range(5):
        print(dataset[1][0])




