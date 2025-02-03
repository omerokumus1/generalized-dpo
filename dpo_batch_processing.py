import pprint
from functools import partial
from typing import List, Optional, Callable

import tiktoken
import torch
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import DataLoader

from PreferenceDataset import PreferenceDataset, DpoPreferenceDataset
from args import Args
from custom_types import DpoBatchEntry, DpoEntryDict, DpoProcessedBatch


def get_max_rejected_length(batch: List[DpoBatchEntry]) -> int:
    return max(len(item["rejected"]) + 1 for item in batch)


def get_max_chosen_length(batch: List[DpoBatchEntry]) -> int:
    return max(len(item["chosen"]) + 1 for item in batch)


def get_max_length_common(batch: List[DpoBatchEntry]) -> int:
    max_length_common = 0
    if batch:
        rejected_max = get_max_rejected_length(batch)
        chosen_max = get_max_chosen_length(batch)
        max_length_common = max(rejected_max, chosen_max, max_length_common)

    return max_length_common


def process_padding_for_chosen(batch_entry: DpoBatchEntry, prompt: Tensor, pad_token_id: int,
                               max_length_common: int,
                               mask_prompt_tokens: bool,
                               processed_batch: DpoProcessedBatch) -> None:
    key = "chosen"

    # Adjust padding according to the common maximum length
    chosen = batch_entry[key]
    pad_length = max_length_common - len(chosen)
    padded = chosen + [pad_token_id] * pad_length
    mask = torch.ones(len(padded)).bool()

    # Set mask for all padding tokens to False
    mask[len(chosen):] = False

    # Set mask for all input tokens to False
    # +2 sets the 2 newline ("\n") tokens before "### Response" to False
    if mask_prompt_tokens:
        mask[:prompt.shape[0] + 2] = False

    processed_batch[key].append(torch.tensor(padded))
    processed_batch["chosen_mask"].append(mask)


def process_padding_for_rejected(batch_entry: DpoBatchEntry, prompt: Tensor, pad_token_id: int,
                                 max_length_common: int,
                                 mask_prompt_tokens: bool,
                                 processed_batch: DpoProcessedBatch):
    key = "rejected"
    # Adjust padding according to the common maximum length
    rejected = batch_entry[key]
    pad_length = max_length_common - len(rejected)
    padded = rejected + [pad_token_id] * pad_length
    mask = torch.ones(len(padded)).bool()

    # Set mask for all padding tokens to False
    mask[len(rejected):] = False

    # Set mask for all input tokens to False
    # +2 sets the 2 newline ("\n") tokens before "### Response" to False
    if mask_prompt_tokens:
        mask[:prompt.shape[0] + 2] = False

    processed_batch[key].append(torch.tensor(padded))
    processed_batch["rejected_mask"].append(mask)


def final_padding_processing_for_chosen(processed_batch: DpoProcessedBatch,
                                        allowed_max_length: Optional[int]):
    for key in ["chosen", "chosen_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(processed_batch[key])  # type: ignore

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        processed_batch[key] = tensor_stack.to(Args.device)  # type: ignore


def final_padding_processing_for_rejected(processed_batch: DpoProcessedBatch,
                                          allowed_max_length: Optional[int]):
    for key in ["rejected", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(processed_batch[key])  # type: ignore

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        processed_batch[key] = tensor_stack.to(Args.device)


def dpo_custom_collate_fn(
        batch: List[DpoBatchEntry],
        pad_token_id=Args.pad_token_id,
        allowed_max_length=None,
        mask_prompt_tokens=True,
) -> DpoProcessedBatch:
    # Initialize lists to hold batch data
    processed_batch: DpoProcessedBatch = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []
    }

    # Determine the longest sequence to set a common padding length

    max_length_common = 0
    if batch:
        max_length_common = get_max_length_common(batch)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        processed_batch["prompt"].append(prompt)

        process_padding_for_chosen(item, prompt, pad_token_id, max_length_common, mask_prompt_tokens,
                                   processed_batch)
        process_padding_for_rejected(item, prompt, pad_token_id, max_length_common, mask_prompt_tokens,
                                     processed_batch)

    # Final processing
    final_padding_processing_for_chosen(processed_batch, allowed_max_length)
    final_padding_processing_for_rejected(processed_batch, allowed_max_length)

    return processed_batch


def get_dpo_customized_collate_fn() -> partial:
    customized_collate_fn = partial(
        dpo_custom_collate_fn,
        mask_prompt_tokens=Args.mask_prompt_tokens,  # This is optional
        allowed_max_length=Args.max_context_length  # The supported context length of the model
    )
    return customized_collate_fn


def test_dpo_customized_collate_fn(collate_fn: partial, data: List[DpoEntryDict], tokenizer,
                                   format_input: Callable[[DpoEntryDict], str]):
    example_data = data[:2]
    print("Example data:")
    print(type(example_data))
    print(type(example_data[0]))
    for i in example_data:
        print()
        pprint.pp(i)

    example_dataset = DpoPreferenceDataset(example_data, tokenizer, format_input)

    example_dataloader = DataLoader(
        example_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=False
    )
    print("Example dataset keys:")
    print(example_dataset[0].keys())

    for batch in example_dataloader:
        print("\n\tbatch.keys:", batch.keys())
        print("\n\tbatch['prompt'].shape:", batch["prompt"].shape)
        print("\n\tbatch['chosen'].shape:", batch["chosen"].shape)
        print("\n\tbatch['chosen_mask'].shape:", batch["chosen_mask"].shape)
        print("\n\tbatch['rejected'].shape:", batch["rejected"].shape)
        print("\n\tbatch['rejected_mask'].shape:", batch["rejecteds_mask"].shape)
        print("\n\tbatch['prompt']:", batch["prompt"])
        print("\n\tbatch['chosen']:", batch["chosen"])
        break
