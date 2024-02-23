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

# https://arxiv.org/abs/2102.11174
class DPFPFeatureMap(nn.Module):
    def __init__(self, head_dim: int, nu: int = 4):
        super().__init__()
        self.nu = nu

    def forward(self, x: torch.Tensor):
        x = torch.cat([x.relu(), -x.relu()], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled

class HadamardFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, head_dim)
        self.layer2 = nn.Linear(head_dim, head_dim)
        
    def forward(self, x: torch.Tensor):        
        return self.layer1(x) * self.layer2(x)    
    

def flatten_outer_product(x, y):
    z = x.unsqueeze(-1) * y.unsqueeze(-2)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    indicies = N * indicies[0] + indicies[1]
    return z.flatten(-2)[..., indicies]

class LearnableOuterProductFeatureMap(nn.Module):
    def __init__(self, head_dim: int, feature_dim: int):
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, feature_dim, bias=False)
        self.layer2 = nn.Linear(head_dim, feature_dim, bias=False)
        self.normalizer = feature_dim ** -0.5
        
    def forward(self, x: torch.Tensor):   
        # x = x * self.normalizer
        return flatten_outer_product(self.layer1(x), self.layer2(x))
