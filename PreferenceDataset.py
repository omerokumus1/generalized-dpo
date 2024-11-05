from typing import List, Callable

from tiktoken import Encoding
from torch.utils.data import Dataset

from custom_types import EntryDict


class PreferenceDataset(Dataset):
    def __init__(self, data: List[EntryDict], tokenizer: Encoding, format_input: Callable[[EntryDict], str]):
        """
        data is the dataset we provided with instruction, input, output, rejecteds, chosen keys
        """
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_responses = entry["rejecteds"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"

            rejected_full_text_list = [
                f"{prompt}\n\n### Response:\n{rejected_response}" for rejected_response in rejected_responses
            ]
            chosen_full_tokens = tokenizer.encode(chosen_full_text)

            rejected_full_tokens_list = [
                tokenizer.encode(rejected_full_text) for rejected_full_text in rejected_full_text_list
            ]

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejecteds": rejected_full_tokens_list,
            })

    def __getitem__(self, index: int):
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)
