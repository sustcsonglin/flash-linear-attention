# -*- coding: utf-8 -*-

import argparse
import time
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from torch.cuda import max_memory_allocated, memory_allocated
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.optimization import get_cosine_schedule_with_warmup

import fla

classes = [getattr(fla.models, i) for i in fla.models.__all__]
configs = {i.model_type: i() for i in classes if issubclass(i, PretrainedConfig)}


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def profile(
    name: str,
    batch_size: int = 8,
    seq_len: int = 2048,
    warmup_steps: int = 16,
    steps: int = 32,
    total_steps: int = 1024,
    lr: float = 3e-4,
    betas: Tuple[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    mixed_precision: str = 'bf16'
):
    device = torch.device('cuda')
    config = configs[name] if name in configs else AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_config(config).cuda().to(dtype)
    num_parameters = model.num_parameters()
    print(f"Initializing {name} model from the config:\n{config}\n{model}")
    print(f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})")
    print(f"Allocated memory after initialization: {sizeof_fmt(memory_allocated(device))}")

    accelerator = Accelerator(mixed_precision=mixed_precision)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    bar = trange(warmup_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    torch.cuda.synchronize(device)
    for _ in bar:
        # forward pass
        tokens = torch.randint(high=config.vocab_size, size=(batch_size, seq_len)).cuda()
        outputs = model(tokens, labels=tokens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        bar.set_description_str(f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}")

    start, total_tokens = time.time(), 0
    bar = trange(steps)
    torch.cuda.synchronize(device)
    for _ in bar:
        # forward pass
        tokens = torch.randint(high=config.vocab_size, size=(batch_size, seq_len), device=device)
        outputs = model(tokens, labels=tokens)
        # backward pass
        accelerator.backward(outputs.loss)
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len
        torch.cuda.synchronize(device)
        duration = time.time() - start
        bar.set_description_str(f"Thoughput: {total_tokens / duration:10.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='retnet')
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seq_len", default=2048, type=int)
    parser.add_argument("--warmup_steps", default=16, type=int)
    parser.add_argument("--steps", default=32, type=int)
    args = parser.parse_args()
    profile(args.name, args.batch_size, args.seq_len, args.warmup_steps, args.steps)
