from fla.modules.utils import checkpoint
import torch.nn as nn
from einops import rearrange
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_fn = None
    causal_conv1d_update = None

@checkpoint
def proj_then_conv1d(x, proj_weight, conv1d_weight, conv1d_bias):
    # We do matmul and transpose BLH -> HBL at the same time
    l = x.shape[-2]
    x = rearrange(
        proj_weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=l,
        )    
    def conv1d(x, conv1d_weight, conv1d_bias):
        return causal_conv1d_fn(
                    x=x,
                    weight=rearrange(conv1d_weight, "d 1 w -> d w"),
                    bias=conv1d_bias,
                    activation="silu",
                )
    y = conv1d(x, conv1d_weight, conv1d_bias)
    return y.transpose(1, 2)

def make_conv1d_module(hidden_size, conv_size, use_bias=False):
    return nn.Conv1d(
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_size=conv_size,
        groups=hidden_size,
        padding=conv_size-1,
        bias=use_bias,
    )
