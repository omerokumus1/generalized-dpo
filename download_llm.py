import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# https://www.youtube.com/watch?v=bwXr8fVOd6Q&ab_channel=DataMagic%28bySunnyKusawa%29
# ? Download and Save the LLM
def download_and_save_model(model_name, path_prefix):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{path_prefix}/tokenizer/{model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(f"{path_prefix}/model/{model_name}")


# ? Load the Saved LLM
# def load_saved_model(model_name, path_prefix):
#     tokenizer = AutoTokenizer.from_pretrained(f"{path_prefix}/tokenizer/{model_name}")
#     model = AutoModelForCausalLM.from_pretrained(f"{path_prefix}/model/{model_name}")
#
#     return model, tokenizer


model_name = "unsloth/Llama-3.2-1B-Instruct"
path_prefix = "cache/llama3_1b"

download_and_save_model(model_name, path_prefix)
