import torch
from supported_llms import LLM


class Args:
    method = "gdpo"
    DEBUG = True
    data_file_path: str = "data/generalized_dpo_data.json"
    finetuned_model_path = "gpt2-medium355M-sft.pth"
    LLM = LLM.unsloth_llama_3_3b_instruct_bnb_4bit
    model_path_prefix = "cache/llama3_1b"
    is_model_local = True

    train_percent: float = 0.85
    test_percent: float = 0.1
    sub_data_size: int = 1000
    use_sub_data = True

    pad_token_id = 128001

    max_seq_length: int = 2048
    max_context_length: int = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 1
    batch_size: int = 4
    mask_prompt_tokens: bool = True

    num_workers: int = 0
    torch_seed: int = 123

    dpo_prompt = """
You're a language model capable of solving multiple-choice questions. Your task is to analyze questions carefully, select the correct answer, and provide a detailed explanation. 
The explanation should justify why the chosen option is correct.

Follow the following rules:
1- Provide only one option as the correct answer.
2- Select the correct answer among the given options and provide a detailed explanation for your choice.
3- Do not ever produce another option like "E) None of the above" or "F) All of the above".
4- Provide your answer only in English and the structure of the answer is as follows:

Question:
[Insert the question here]

Options:
[Insert the options here, e.g., A) Option A\nB) Option B]

Correct Answer:
[Insert the correct option here, e.g., A) Option A]

Explanation:
[Provide a detailed explanation in English, explaining why the selected option is correct.]

Here is an example:

Question:
Which of the following sentences uses the correct form of the verb in the present perfect tense?

Options:
A) She has visited Paris last year.
B) He has already finished his homework.

Correct Answer:
B) He has already finished his homework.

Explanation:
- The present perfect tense is formed with "has" or "have" + past participle of the main verb.
- The sentence "He has already finished his homework" is grammatically correct because:
\t- "has" is used for third-person singular (he).
\t- "finished" is the correct past participle of the verb "finish."
\t- The adverb "already" is correctly placed between "has" and "finished," enhancing the meaning without changing the structure.

Now, answer the following question in the same format:"""

    gdpo_prompt = """
You're a language model capable of solving multiple-choice questions. Your task is to analyze questions carefully, select the correct answer, and provide a detailed explanation. 
The explanation should justify why the chosen option is correct.

Follow the following rules:
1- Provide only one option as the correct answer.
2- Select the correct answer among the given options and provide a detailed explanation for your choice.
3- Do not ever produce another option like "E) None of the above" or "F) All of the above".
4- Provide your answer only in English and the structure of the answer is as follows:

Question:
[Insert the question here]

Options:
[Insert the options here, e.g., A) Option A\nB) Option B\nC) Option C\nD) Option D]

Correct Answer:
[Insert the correct option here, e.g., A) Option A]

Explanation:
[Provide a detailed explanation in English, explaining why the selected option is correct.]

Here is an example:

Question:
Which of the following sentences uses the correct form of the verb in the present perfect tense?

Options:
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
