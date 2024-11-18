# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

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

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizer
    varlen: bool = False
    return_tensors: str = "pt"

    def __call__(
        self,
        examples: List[Union[List[int], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if not isinstance(examples[0], Dict):
            examples = [{'input_ids': x} for x in examples]
        if isinstance(examples[0]['input_ids'], List):
            examples = [{'input_ids': torch.tensor(x['input_ids'], dtype=torch.long)} for x in examples]
        elif isinstance(examples[0]['input_ids'], np.ndarray):
            examples = [{'input_ids': torch.from_numpy(x['input_ids'])} for x in examples]

        if not self.varlen:
            length_of_first = examples[0]['input_ids'].size(0)
            # Check if padding is necessary.
            if all(x['input_ids'].size(0) == length_of_first for x in examples):
                batch = {'input_ids': torch.stack([x['input_ids'] for x in examples], dim=0)}
            else:
                # If yes, check if we have a `pad_token`.
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        f"You are attempting to pad samples but the tokenizer you are using "
                        f"({self.tokenizer.__class__.__name__}) does not have a pad token."
                    )
                batch = self.tokenizer.pad(examples, return_tensors=self.return_tensors, return_attention_mask=False)
        else:
            batch = {'input_ids': torch.cat([x['input_ids'] for x in examples], dim=0).unsqueeze(0)}
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
