import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class CSVDataset(Dataset):

    def __init__(self, filepath, target_column):
        df = pd.read_csv(filepath)
        self.X = torch.tensor(data=df.drop(target_column, axis=1).values, dtype=torch.float32)
        self.Y = torch.tensor(data=df[target_column].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

