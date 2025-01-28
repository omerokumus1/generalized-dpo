import pprint
from functools import partial
from typing import List, Optional, Callable

import tiktoken
import torch
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import DataLoader

from PreferenceDataset import PreferenceDataset
from args import Args
from custom_types import BatchEntry, ProcessedBatch, EntryDict


def get_max_rejecteds_length(batch: List[BatchEntry]) -> int:
    rejecteds_len = []
    for item in batch:
        rejecteds_len.append(max(len(r) + 1 for r in item["rejecteds"]))

    return max(rejecteds_len)


def get_max_chosen_length(batch: List[BatchEntry]) -> int:
    return max(len(item["chosen"]) + 1 for item in batch)


def get_max_length_common(batch: List[BatchEntry]) -> int:
    max_length_common = 0
    if batch:
        rejecteds_max = get_max_rejecteds_length(batch)
        chosen_max = get_max_chosen_length(batch)
        max_length_common = max(rejecteds_max, chosen_max, max_length_common)

    return max_length_common


def process_padding_for_chosen(batch_entry: BatchEntry, prompt: Tensor, pad_token_id: int,
                               max_length_common: int,
                               mask_prompt_tokens: bool,
                               processed_batch: ProcessedBatch) -> None:
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


def process_padding_for_rejecteds(batch_entry: BatchEntry, prompt: Tensor, pad_token_id: int,
                                  max_length_common: int,
                                  mask_prompt_tokens: bool,
                                  processed_batch: ProcessedBatch):
    key = "rejecteds"
    padded_rejecteds = []
    padded_rejecteds_mask = []
    for rejected in batch_entry[key]:
        # Adjust padding according to the common maximum length
        pad_length = max_length_common - len(rejected)
        padded = rejected + [pad_token_id] * pad_length
        mask = torch.ones(len(padded)).bool()

        # Set mask for all padding tokens to False
        mask[len(rejected):] = False

        # Set mask for all input tokens to False
        # +2 sets the 2 newline ("\n") tokens before "### Response" to False
        if mask_prompt_tokens:
            mask[:prompt.shape[0] + 2] = False

        padded_rejecteds.append(torch.tensor(padded))
        padded_rejecteds_mask.append(mask)

    processed_batch[key].append(padded_rejecteds)
    processed_batch["rejecteds_mask"].append(padded_rejecteds_mask)


def final_padding_processing_for_chosen(processed_batch: ProcessedBatch,
                                        allowed_max_length: Optional[int]):
    for key in ["chosen", "chosen_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(processed_batch[key])  # type: ignore

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        processed_batch[key] = tensor_stack.to(Args.device)  # type: ignore


def final_padding_processing_for_rejecteds(processed_batch: ProcessedBatch,
                                           allowed_max_length: Optional[int]):
    for key in ["rejecteds", "rejecteds_mask"]:
        # outer_list is  List[List[Tensor]]
        outer_list = processed_batch[key]  # type: ignore
        for i in range(len(outer_list)):
            # Stack all sequences into a tensor for the given key
            tensor_stack = torch.stack(outer_list[i])

            # Optionally truncate to maximum sequence length
            if allowed_max_length is not None:
                tensor_stack = tensor_stack[:, :allowed_max_length]

            # Move to the specified device
            # outer_list becomes Tensor with shape (num_rejecteds, max_length)
            outer_list[i] = tensor_stack.to(Args.device)


def gdpo_custom_collate_fn(
        batch: List[BatchEntry],
        pad_token_id=Args.pad_token_id,
        allowed_max_length=None,
        mask_prompt_tokens=True,
) -> ProcessedBatch:
    # Initialize lists to hold batch data
    processed_batch: ProcessedBatch = {
        "prompt": [],
        "chosen": [],
        "rejecteds": [],
        "rejecteds_mask": [],
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
        process_padding_for_rejecteds(item, prompt, pad_token_id, max_length_common, mask_prompt_tokens,
                                      processed_batch)

    # Final processing
    final_padding_processing_for_chosen(processed_batch, allowed_max_length)
    final_padding_processing_for_rejecteds(processed_batch, allowed_max_length)

    return processed_batch


def get_customized_collate_fn() -> partial:
    customized_collate_fn = partial(
        gdpo_custom_collate_fn,
        mask_prompt_tokens=Args.mask_prompt_tokens,  # This is optional
        allowed_max_length=Args.max_context_length  # The supported context length of the model
    )
    return customized_collate_fn


def test_customized_collate_fn(collate_fn: partial, data: List[EntryDict], tokenizer,
                               format_input: Callable[[EntryDict], str]):
    example_data = data[:2]
    print("Example data:")
    print(type(example_data))
    print(type(example_data[0]))
    for i in example_data:
        print()
        pprint.pp(i)

    example_dataset = PreferenceDataset(example_data, tokenizer, format_input)

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
        print("\n\tbatch['rejecteds'][0].shape:", batch["rejecteds"][0].shape)
        print("\n\tbatch['rejecteds_mask'][0].shape:", batch["rejecteds_mask"][0].shape)
        print("\n\tbatch['prompt']:", batch["prompt"])
        print("\n\tbatch['chosen']:", batch["chosen"])
        break


def decode_tokens_from_batch(token_ids, tokenizer) -> str:
    ids_in_python_list = token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list, skip_special_tokens=True)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist(), skip_special_tokens=True)


def test_decode_tokens_from_batch(data_loader: DataLoader, tokenizer: Encoding):
    for batch in data_loader:
        print("batch.keys:", batch.keys())
        print("batch['prompt']:", batch["prompt"])
        print("First propt's token ids")
        print(batch["prompt"][0])
        first_prompt_decoded = decode_tokens_from_batch(
            token_ids=batch["prompt"][0],  # [0] for the first entry in the batch
            tokenizer=tokenizer,
        )
        print("\nFirst prompt decoded from token ids")
        print(first_prompt_decoded)
        break


def extract_response(response_text: str, input_text: str) -> str:
    return response_text[len(input_text):].replace("### Response:", "").strip()
