import json
import pprint
from typing import Tuple, List

from args import Args
from custom_types import EntryDict


def read_data(file_path: str) -> List[EntryDict]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def print_data(data: List[EntryDict], starting: int = 0, ending: int = 3):
    pprint.pp(data[starting:ending])


def get_sub_data(data: List[EntryDict],
                 starting: int = 0,
                 ending: int = Args.sub_data_size):
    return data[starting:ending]


def format_input(entry: EntryDict) -> str:
    instruction_text = (
        f"Aşağıda bir soru ve bu soru için doğru olabilecek seçenekler A), B), C) şeklinde girdi olarak verilmiştir. "
        f"Verilen soruya göre doğru cevabı seç ve açıkla. "
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def get_desired_response_by_entry(entry: EntryDict) -> str:
    return f"### Response:\n{entry['chosen']}"


def get_desired_response_by_index(data: List[EntryDict], index: int) -> str:
    return f"### Response:\n{data[index]['chosen']}"


def get_possible_responses_by_entry(entry: EntryDict) -> str:
    return f"### Responses:\n{str(entry['rejecteds'])}"


def get_possible_responses_by_index(data: List[EntryDict], index: int) -> str:
    return f"### Responses:\n{str(data[index]['rejecteds'])}"


def get_train_test_validation_data(data: List[EntryDict], train_percent: float, test_percent: float) -> Tuple[
    List[EntryDict], List[EntryDict], List[EntryDict]]:
    train_portion = int(len(data) * train_percent)
    test_portion = int(len(data) * test_percent)
    val_portion = len(data) - train_portion - test_portion  # Remaining percent
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, test_data, val_data


def print_data_lengths(data: List[EntryDict], train_data: List[EntryDict], test_data: List[EntryDict],
                       val_data: List[EntryDict]):
    print("\tData length:", len(data))
    print("\tTraining set length:", len(train_data))
    print("\tTest set length:", len(test_data))
    print("\tValidation set length:", len(val_data))
