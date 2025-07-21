import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MyNet(nn.Module):
    def __init__(self, input_dim : int, output_size : int, layers_size : List[int], dropout : float, layernorm : bool):
         # layernorm, rete parametrica, dropout parametrico, test con 1 attacco e benign
        super().__init__()
        
        layers = []

        input_size = input_dim

        for layer_size in layers_size:

            # Hidden Layer
            layers.append(nn.Linear(input_size, layer_size))

            # Requested LayerNorm
            if layernorm:
                layers.append(nn.LayerNorm(layer_size))

            # Activator
            layers.append(nn.ReLU())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Update layer input
            input_size = layer_size
        
        # Output logits
        layers.append(nn.Linear(input_size, output_size))

        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)