from enum import Enum


class LLM(Enum):
    unsloth_llama_3_8b_bnb_4bit = "unsloth/llama-3-8b-bnb-4bit"
    unsloth_llama_3_3b_instruct_bnb_4bit = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    unsloth_llama_3_3b_instruct = "unsloth/Llama-3.2-3B-Instruct"
    unsloth_llama_3_1b_bnb_4bit = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    # ytu_turkish_gpt2_large = "ytu-ce-cosmos/turkish-gpt2-large"
    ytu_turkish_llama_8b_v01 = "ytu-ce-cosmos/Turkish-Llama-8b-v0.1"
    ytu_turkish_llama_8b_instruct_v01 = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
    unsloth_llama3_8b_instruct = "unsloth/Meta-Llama-3.1-8B-Instruct"
    unsloth_llama3_1b_instruct = "unsloth/Llama-3.2-1B-Instruct"
