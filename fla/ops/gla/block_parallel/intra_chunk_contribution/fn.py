# -*- coding: utf-8 -*-

import torch

from .fn_only_gk import IntraCalA
from .fn_only_gv import IntraCalO


def intra_chunk_onc(q, k, v, gk, gv):
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    if gk is not None:
        assert gk.is_contiguous()
    if gv is not None:
        assert gv.is_contiguous()

    assert k.shape[-2] % 16 == 0

    if gk is not None:
        A = IntraCalA.apply(q, k, gk)
    else:
        A = q @ k.transpose(-1, -2)

    mask = torch.triu(torch.ones(A.shape[-2], A.shape[-2]), diagonal=1).bool().to(A.device)
    A.masked_fill_(mask, 0)

    return IntraCalO.apply(A, v, gv) if gv is not None else A.to(v) @ v
