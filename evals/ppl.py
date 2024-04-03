# -*- coding: utf-8 -*-

import argparse
import math
from functools import partial
from itertools import chain
from typing import Any, Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer)

import fla  # noqa


def preprocess(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
    max_len: int = 2048
) -> Dict[str, List[List[int]]]:
    examples = [text + tokenizer.eos_token for text in examples['text']]
    tokenized_examples = tokenizer(examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // max_len) * max_len
    result = {
        k: [t[i: i + max_len] for i in range(0, total_length, max_len)]
        for k, t in concatenated_examples.items()
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('-d', '--data', type=str, default='EleutherAI/lambada_openai')
    parser.add_argument('-s', '--split', type=str, default='test')
    parser.add_argument('--max_len', type=str, default=2048)
    parser.add_argument('--block_size', type=str, default=128)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device},
        torch_dtype=dtype
    )
    model.eval()
    print(f"{model}")

    dataset = load_dataset(args.data, split=args.split)
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_len=args.max_len),
        batched=True,
        remove_columns=['text']
    )
    bar = tqdm(dataset)

    with torch.no_grad():
        total_loss = 0
        block_loss = [0 for _ in range(0, args.max_len, args.block_size)]
        loss_fct = nn.CrossEntropyLoss()
        for n, sentence in enumerate(bar, 1):
            input_ids = torch.tensor([sentence['input_ids']]).to(device, dtype=torch.long)
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs['loss'], outputs['logits']
            total_loss += loss.item()
            labels = torch.cat((input_ids[..., 1:], torch.full_like(input_ids[:, :1], loss_fct.ignore_index)), 1)
            for i, j in enumerate(range(0, args.max_len, args.block_size)):
                block_loss[i] += loss_fct(logits[:, j:j+args.block_size].view(-1, model.config.vocab_size),
                                          labels[:, j:j+args.block_size].view(-1))

            bar.set_description_str(f"Perplexity: {math.exp(total_loss / n):.4f}")
    print("Loss breakdown by length:")
    for i, loss in enumerate(block_loss):
        print(f"{i*args.block_size:>4} - {min(i*args.block_size+args.block_size, args.max_len):>4}: "
              f"{math.exp(loss / len(dataset)):.4f}")
