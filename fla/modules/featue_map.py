import torch
from torch import nn


# https://arxiv.org/abs/2402.04347
class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()
        
    def init_weights_(self):
        """Initialize trainable map as identity"""
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):        
        x = self.layer(x)  # shape b, h, l, d        
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)
       

# https://arxiv.org/abs/2103.13076
class T2RFeatureMap(nn.Module):
    def __init__(self, head_dim: int, dot_dim: int = None):
        super().__init__()
        # Trainable map
        if dot_dim is None:
            dot_dim = head_dim
        self.layer = nn.Linear(head_dim, dot_dim)
        
    def forward(self, x: torch.Tensor):
        return self.layer(x).relu()  
        
