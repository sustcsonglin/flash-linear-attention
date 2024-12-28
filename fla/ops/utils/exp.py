# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

import triton
import triton.language as tl


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float('-inf')))
