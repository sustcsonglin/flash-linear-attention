# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang
# Gated Linear Attention Transformers with Hardware-Efficient Training: https://arxiv.org/abs/2312.06635
# chunkwise block parallel. Materialize chunkwise hidden states into HBMs.
# Therefore it is neccessary to have a large chunk size to reduce such materialization overhead.

import torch.nn.functional as F
from einops import rearrange

from fla.ops.triton.gla.block_parallel.inter_chunk_contribution.fn import \
    inter_chunk_onc
from fla.ops.triton.gla.block_parallel.intra_chunk_contribution.fn import \
    intra_chunk_onc


def pad_and_rearrange(x, chunk_size):
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, chunk_size - x.shape[-2] % chunk_size))
    if x.shape[-1] % 32 != 0:
        x = F.pad(x, (0, 32 - x.shape[-1] % 32))
    x = rearrange(x, '... (n c) d -> ... n c d', c=chunk_size)
    return x


def chunk_gla(q, k, v, gk=None, gv=None, chunk_size=128):
    scale = (q.shape[-1])**-0.5
    seq_len = q.shape[-2]
    output_dim = v.shape[-1]
    q, k, v = map(lambda x: pad_and_rearrange(x, chunk_size), [q, k, v])
    q = q * scale
    if gk is not None:
        gk = pad_and_rearrange(gk, chunk_size)
    if gv is not None:
        gv = pad_and_rearrange(gv, chunk_size)
    gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv)
    o2 = intra_chunk_onc(q, k, v, gk, gv)
    o = rearrange(o1+o2, 'b h n c d -> b h (n c) d')
    return o[:, :, :seq_len, :output_dim]
