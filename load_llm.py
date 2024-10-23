import tiktoken
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel

from args import Args
from enum import Enum


class LLM(Enum):
    unsloth_llama_3_8b_bnb_4bit = "unsloth/llama-3-8b-bnb-4bit"
    gpt2_small = "gpt2-small"
    gpt2_medium = "gpt2-medium"
    gpt2_large = "gpt2-large"
    gpt2_xl = "gpt2-xl"


def load_llm(llm_name: LLM):
    if llm_name == LLM.unsloth_llama_3_8b_bnb_4bit:
        max_seq_length = Args.max_seq_length  # Choose any! We auto support RoPE Scaling internally!
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
        return FastLanguageModel.from_pretrained(
            model_name=llm_name.value,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
