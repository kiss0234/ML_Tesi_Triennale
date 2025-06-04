import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):
    
    def __init__(self, input, output):
        super().__init__()
        self.fc1 = nn.Linear(input, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x