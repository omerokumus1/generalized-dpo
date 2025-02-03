import json
import pprint
from typing import Tuple, List

from args import Args
from custom_types import EntryDict, DpoEntryDict


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


def format_input(entry: EntryDict | DpoEntryDict) -> str:
    instruction_text = ("""
You're a language model capable of solving multiple-choice questions. Your task is to analyze questions carefully, select the correct answer, and provide a detailed explanation. 
The explanation should justify why the chosen option is correct and why the others are incorrect.

Do not ever produce another option like "E) None of the above" or "F) All of the above".
Provide only one option as the correct answer.
Provide your answer only in English and the structure of the answer is as follows:

Question:
[Insert the question here]

Correct Answer:
[Insert the correct option here, e.g., A)]

Explanation:
[Provide a detailed explanation in English, explaining why the selected option is correct and the others are incorrect.]

Here is an example:

Question:
Which of the following sentences uses the correct form of the verb in the present perfect tense?

A) She has visited Paris last year.
B) They have been to the park yesterday.
C) He has already finished his homework.
D) We has gone to the store earlier.

Correct Answer:
C) He has already finished his homework.

Explanation:
- The present perfect tense is formed with "has" or "have" + past participle of the main verb.
- The sentence "He has already finished his homework" is grammatically correct because:
\t- "has" is used for third-person singular (he).
\t- "finished" is the correct past participle of the verb "finish."
\t- The adverb "already" is correctly placed between "has" and "finished," enhancing the meaning without changing the structure.

Now, answer the following question in the same format:"""
                        + f"\n\n### Instruction:\n{entry['instruction']}")

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


def get_train_test_validation_data(data, train_percent: float, test_percent: float):
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
