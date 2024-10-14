# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
from itertools import chain
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    context_length: int
) -> Dict[str, List[List[int]]]:
    """
    Tokenize the input text and split into chunks of specified context length.

    Args:
        examples:
            Dictionary containing the input text.
        tokenizer:
            Initialized tokenizer.
        context_length:
            Length of each context chunk.

    Returns:
        Dictionary containing tokenized and chunked input ids
    """
    text = examples['text']
    input_ids = tokenizer(text)['input_ids']
    input_ids = list(chain(*input_ids))
    total_length = len(input_ids)
    total_length = (total_length // context_length) * context_length
    # The last chunk smaller than context_length will be discarded
    return {'input_ids': [input_ids[i:i+context_length] for i in range(0, total_length, context_length)]}


def preprocess(
    dataset: str,
    name: Optional[str] = None,
    split: str = 'train',
    output: str = 'data',
    model: str = 'mistralai/Mistral-7B-v0.1',
    num_proc: int = 64,
    context_length: int = 8192
) -> None:
    """
    Load, tokenize, and save the processed dataset.

    Args:
        dataset:
            Path or name of the dataset.
        name:
            Name of the dataset configuration.
        split:
            Dataset split to process.
        output:
            Output directory.
        model:
            Model name for tokenizer.
        num_proc:
            Number of processes for parallel processing.
        context_length:
            Context length for tokenization.
    """
    tokenized_path = f'{output}/{dataset}/{name}/{split}' if name is not None else f'{output}/{dataset}/{split}'

    logging.info(f'Initializing tokenizer of {model}')
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    logging.info(f'Tokenizer initialized: {tokenizer}')

    logging.info(f'Loading dataset: {dataset}')
    dataset = load_dataset(dataset, name=name, split=split)

    remove_columns = list(next(iter(dataset)).keys())
    logging.info('Tokenizing and processing dataset')
    dataset = dataset.map(
        lambda examples: tokenize(examples, tokenizer, context_length),
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Running tokenizer on dataset"
    )

    logging.info(f'Saving processed dataset to {tokenized_path}')
    dataset.save_to_disk(tokenized_path, num_proc=num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", help="Path or name of the dataset")
    parser.add_argument("--name", default="sample-10BT", help="Name of the dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name for tokenizer")
    parser.add_argument("--num_proc", type=int, default=64, help="Number of processes for parallel processing")
    parser.add_argument("--context_length", type=int, default=8192, help="Context length for tokenization")
    args = parser.parse_args()

    preprocess(
        dataset=args.dataset,
        name=args.name,
        split=args.split,
        output=args.output,
        model=args.model,
        num_proc=args.num_proc,
        context_length=args.context_length
    )
