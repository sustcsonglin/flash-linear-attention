# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        varlen (`bool`):
            Whether to return sequences with variable lengths.
            If `True`, the offsets indicating the start and end of each sequence will be returned.
            For example, if the sequence lengths are `[4, 8, 12]`,
            the returned `input_ids` will be a long flattened tensor of shape `[1, 24]`, with `offsets` being `[0, 4, 12, 24]`.
            If `False`, the `input_ids` with shape `[batch_size, seq_len]` will be returned directly.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "pt".
    """

    tokenizer: PreTrainedTokenizer
    varlen: bool = False
    return_tensors: str = "pt"

    def __call__(
        self,
        examples: List[Union[List[int], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if not isinstance(examples[0], Dict):
            examples = [{'input_ids': example} for example in examples]

        def tensorize(example: Dict[str, Any]) -> Dict[str, Any]:
            tensorized = {}
            for key in ['input_ids', 'offsets']:
                if key not in example:
                    continue
                if isinstance(example[key], List):
                    tensorized[key] = torch.tensor(example[key], dtype=torch.long)
                elif isinstance(example[key], np.ndarray):
                    tensorized[key] = torch.from_numpy(example[key])
            return tensorized

        examples = list(map(tensorize, examples))

        if not self.varlen:
            length_of_first = examples[0]['input_ids'].size(0)
            # Check if padding is necessary.
            if all(example['input_ids'].size(0) == length_of_first for example in examples):
                batch = {
                    'input_ids': torch.stack([example['input_ids'] for example in examples], dim=0),
                }
            else:
                # If yes, check if we have a `pad_token`.
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        f"You are attempting to pad samples but the tokenizer you are using "
                        f"({self.tokenizer.__class__.__name__}) does not have a pad token."
                    )
                batch = self.tokenizer.pad(examples, return_tensors=self.return_tensors, return_attention_mask=False)
        else:
            if len(examples) > 1:
                raise ValueError("The batch size must be 1 for variable length inputs.")
            batch = {
                'input_ids': torch.cat([example['input_ids'] for example in examples], dim=0).unsqueeze(0)
            }
            if 'offsets' in examples[0]:
                batch['offsets'] = torch.cat([example['offsets'] for example in examples], dim=0).unsqueeze(0)
            else:
                # determine boundaries by bos/eos positions
                if self.tokenizer.add_bos_token:
                    offsets = []
                    if batch['input_ids'][0, 0] != self.tokenizer.bos_token_id:
                        offsets.append(torch.tensor([0], dtype=torch.long))
                    offsets.append(torch.where(batch['input_ids'].eq(self.tokenizer.bos_token_id))[1])
                    offsets.append(torch.tensor([len(batch['input_ids'][0])], dtype=torch.long))
                    batch['offsets'] = torch.cat(offsets, dim=0)
                elif self.tokenizer.add_eos_token:
                    offsets = [torch.tensor([0], dtype=torch.long)]
                    offsets.append(torch.where(batch['input_ids'].eq(self.tokenizer.eos_token_id))[1] + 1)
                    if batch['input_ids'][0, -1] != self.tokenizer.eos_token_id:
                        offsets.append(torch.tensor([len(batch['input_ids'][0])], dtype=torch.long))
                    batch['offsets'] = torch.cat(offsets, dim=0)
                else:
                    raise ValueError("You must allow the tokenizer to add either a bos or eos token as separators.")

        labels = batch['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
