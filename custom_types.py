from typing import TypedDict, List

import torch
from torch import Tensor


class EntryDict(TypedDict):
    instruction: str
    input: str
    output: str
    chosen: str
    rejecteds: List[str]

class DpoEntryDict(TypedDict):
    instruction: str
    input: str
    output: str
    chosen: str
    rejected: str


class ProcessedBatch(TypedDict):
    prompt: List[Tensor]  # List of tensor
    chosen: List[Tensor]  # Finally becomes Tensor with shape (batch_size, max_length)
    rejecteds: List[List[Tensor]]  # Finally becomes Tensor with shape (batch_size, num_rejecteds, max_length)
    rejecteds_mask: List[List[Tensor]]  # Finally becomes Tensor with shape (batch_size, num_rejecteds, max_length)
    chosen_mask: List[Tensor]  # Finally becomes Tensor with shape (batch_size, max_length)

class DpoProcessedBatch(TypedDict):
    prompt: List[Tensor]
    chosen: List[Tensor]
    rejected: List[Tensor]
    rejected_mask: List[Tensor]  # Finally becomes Tensor with shape (batch_size, num_rejecteds, max_length)
    chosen_mask: List[Tensor]  # Finally becomes Tensor with shape (batch_size, max_length)


class BatchEntry(TypedDict):
    prompt: List[int]
    chosen: List[int]
    rejecteds: List[List[int]]


class DpoBatchEntry(TypedDict):
    prompt: List[int]
    chosen: List[int]
    rejected: List[int]
