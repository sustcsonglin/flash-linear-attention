# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang
import triton
import triton.language as tl


@triton.jit
def safe_diff_exp(g, chunk_size, q_first=True):
    if q_first:
        b_D = g[:, None] - g[None, :]
        b_D = tl.where(b_D <= 0, b_D, float('-inf'))
        b_D = tl.where(tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :], b_D, float('-inf'))
    else:
        b_D = g[None, :] - g[:, None]
        b_D = tl.where(b_D <= 0, b_D, float('-inf'))
        b_D = tl.where(tl.arange(0, chunk_size)[:, None] <= tl.arange(0, chunk_size)[None, :], b_D, float('-inf'))
    return tl.exp(b_D)
