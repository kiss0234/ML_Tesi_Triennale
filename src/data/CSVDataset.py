import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class CSVDataset(Dataset):

    def __init__(self, target_column, filepath = None, data = None):
        if filepath is not None:
            df = pd.read_csv(filepath)
        elif data is not None:
            df = data
        else:
            raise ValueError("Either 'filepath' or 'data' must be provided.")

        self.X = torch.tensor(data=df.drop(target_column, axis=1).values, dtype=torch.float32)
        self.Y = torch.tensor(data=df[target_column].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

