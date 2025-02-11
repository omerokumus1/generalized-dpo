from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from args import Args
from evaluating import get_model_responses
from load_llm import load_llm
from prepare_dataset import read_data, get_train_test_validation_data
from utils import extract_answers, write_to_txt, write_to_json

Args.data_file_path = "gdpo_data_en.json"

gdpo_model, tokenizer = load_llm("BIGDaTA-Lab/Llama-3.2-1B-4bit-generalized-dpo")
dpo_model, _ = load_llm("BIGDaTA-Lab/Llama-3.2-1B-4bit-dpo")
reference_model, _ = load_llm("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")

gdpo_model.eval()
dpo_model.eval()
reference_model.eval()

data = read_data(Args.data_file_path)
train_data, test_data, val_data = get_train_test_validation_data(data, Args.train_percent, Args.test_percent)

gdpo_responses, dpo_responses, reference_responses, correct_responses, inputs = get_model_responses(gdpo_model, dpo_model, reference_model, val_data, tokenizer, response_count=10)

model_responses = {
    "gdpo_responses": gdpo_responses,
    "dpo_responses": dpo_responses,
    "reference_responses": reference_responses,
    "correct_responses": correct_responses
}

write_to_json(model_responses, "inference/all_models_responses.json")

gdpo_answers = extract_answers(gdpo_responses)
dpo_answers = extract_answers(dpo_responses)
reference_answers = extract_answers(reference_responses)
correct_answers = extract_answers(correct_responses)

answers = {
    "gdpo_answers": gdpo_answers,
    "dpo_answers": dpo_answers,
    "reference_answers": reference_answers,
    "correct_answers": correct_answers
}

write_to_txt(answers, "inference/all_models_answers.txt")