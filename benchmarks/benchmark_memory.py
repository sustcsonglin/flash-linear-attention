# -*- coding: utf-8 -*-

import argparse

import torch
from torch.cuda import max_memory_allocated
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from fla.models.abc import ABCConfig, ABCForCausalLM
from fla.models.deltanet import DeltaNetConfig, DeltaNetForCausalLM
from fla.models.gla import GLAConfig, GLAForCausalLM
from fla.models.linear_attn import (LinearAttentionConfig,
                                    LinearAttentionForCausalLM)
from fla.models.retnet import RetNetConfig, RetNetForCausalLM

configs = {
    'abc': ABCConfig,
    'deltanet': DeltaNetConfig,
    'gla': GLAConfig,
    'linear_attn': LinearAttentionConfig,
    'retnet': RetNetConfig,
}
models = {
    'abc': ABCForCausalLM,
    'deltanet': DeltaNetForCausalLM,
    'gla': GLAForCausalLM,
    'linear_attn': LinearAttentionForCausalLM,
    'retnet': RetNetForCausalLM,
}


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def profile_memory(name: str, batch_size: int = 8, seq_len: int = 2048, steps: int = 10):
    device = torch.device('cuda')

    config = configs[name]()
    model = AutoModelForCausalLM.from_config(config).cuda()
    print(f"Initializing {name} model from the config")
    print(config)
    print(model)
    print(f"Allocated memory after initialization: {sizeof_fmt(max_memory_allocated(device))}")

    optimizer = AdamW(model.parameters())

    for _ in tqdm(range(steps)):
        # create random inputs

        # forward pass
        tokens = torch.randint(high=config.vocab_size, size=(batch_size, seq_len)).cuda()
        outputs = model(tokens, labels=tokens)
        # backward pass
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='gla')
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--steps", default=10, type=int)
    args = parser.parse_args()

    profile_memory(args.name, args.batch_size, args.seq_len, args.steps)
