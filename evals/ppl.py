# -*- coding: utf-8 -*-

import argparse
import math
from functools import partial
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer)

from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss


def preprocess(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, List[List[int]]]:
    examples = tokenizer(examples['text'])
    examples = {
        'input_ids': examples['input_ids'],
        'length': [len(input_ids) for input_ids in examples['input_ids']]
    }
    return examples


def batchify(dataset, tokens_per_batch):
    count, batch = 0, []
    for sentence in dataset.sort('length'):
        count += sentence['length']
        batch.append(sentence)
        if count >= tokens_per_batch:
            yield [i['input_ids'] for i in batch]
            count, batch = 0, []
    if len(batch) > 0:
        yield [i['input_ids'] for i in batch]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument('-p', '--path', type=str, default='fla-hub/gla-1.3B-100B')
    parser.add_argument('-d', '--data', type=str, default='fla-hub/slimpajama-test')
    parser.add_argument('-s', '--split', type=str, default='test')
    parser.add_argument('--max_len', type=int, default=32000)
    parser.add_argument('--block_size', type=int, default=2048)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float
    torch.manual_seed(0)

    print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(args.path, device_map={"": device}, torch_dtype=dtype)
    model.eval()
    print(f"{model}")

    dataset = load_dataset(args.data, split='train')
    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True)

    with torch.no_grad():
        total_loss = 0
        total_tokens = 0
        total_sentences = 0
        block_loss = [torch.tensor(0., dtype=torch.float, device=device) for _ in range(0, args.max_len, args.block_size)]
        block_tokens = [1e-10 for _ in range(0, args.max_len, args.block_size)]
        loss_fct = FusedCrossEntropyLoss(reduction='sum')
        bar = tqdm(batchify(dataset, 8 * 2048))
        for batch in bar:
            input_ids = pad_sequence(sequences=[torch.tensor(i, dtype=torch.long, device=device) for i in batch],
                                     batch_first=True,
                                     padding_value=tokenizer.eos_token_id)
            labels = torch.where(input_ids.eq(tokenizer.eos_token_id), loss_fct.ignore_index, input_ids)
            outputs = model(input_ids, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']
            labels = torch.cat((input_ids[..., 1:], torch.full_like(input_ids[:, :1], tokenizer.eos_token_id)), -1)
            nlls = (-logits.log_softmax(-1)).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            labels = torch.where(labels.eq(tokenizer.eos_token_id), loss_fct.ignore_index, labels)
            nlls = torch.where(labels.eq(loss_fct.ignore_index), 0., nlls)

            total_loss += loss.item() * labels.ne(loss_fct.ignore_index).sum()
            total_tokens += labels.ne(loss_fct.ignore_index).sum()
            total_sentences += input_ids.shape[0]

            for i, j in enumerate(range(0, min(input_ids.shape[-1], args.max_len), args.block_size)):
                block_loss[i] += nlls[:, j:j+args.block_size].sum()
                block_tokens[i] += labels[:, j:j+args.block_size].ne(loss_fct.ignore_index).sum()
            ppls = [f"{math.exp(loss / toks):6.2f}" for loss, toks in zip(block_loss, block_tokens)]
            bar.set_description_str(f"{total_tokens} tokens, {total_sentences} sentences: " + ' '.join(ppls))
