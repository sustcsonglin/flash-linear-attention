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
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "pt".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizer
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
        labels = batch['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
