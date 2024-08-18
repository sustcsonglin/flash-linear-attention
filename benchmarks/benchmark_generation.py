# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa


def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--path", type=str, default="fla-hub/transformer-1.3B-100B")
    parser.add_argument("--prompt", type=str, default="Hello everyone, I'm Songlin Yang")
    parser.add_argument("--maxlen", type=int, default=64)
    parser.add_argument("--cache", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Loading {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"{tokenizer}")

    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype,
        use_cache=args.cache
    )
    model.eval()
    print(f"{model}")
    print(f"Number of parameters: {sizeof_fmt(model.num_parameters())}\n")

    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    max_length = input_ids.shape[1] + args.maxlen

    torch.cuda.synchronize()
    start = time.time()
    text = model.generate(
        input_ids=input_ids,
        use_cache=args.cache,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty
    )
    print(f"Prompt:\n{args.prompt}")
    print(f"Generated:\n{tokenizer.batch_decode(text, skip_special_tokens=True)[0]}\n")
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(text[0]) - len(input_ids[0])}")
    print(f"Total prompt processing + decoding time: {elapsed * 1000:.0f}ms")
    print(f"Max memory used: {sizeof_fmt(torch.cuda.max_memory_allocated())}")
