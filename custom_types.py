from typing import TypedDict, List

from torch import Tensor


class EntryDict(TypedDict):
    instruction: str
    input: str
    output: str
    chosen: str
    rejecteds: List[str]


class ProcessedBatch(TypedDict):
    prompt: List[Tensor]
    chosen: List[Tensor]
    rejecteds: List[List[Tensor]]
    rejecteds_mask: List[List[Tensor]]
    chosen_mask: List[Tensor]


class BatchEntry(TypedDict):
    prompt: List[int]
    chosen: List[int]
    rejecteds: List[List[int]]