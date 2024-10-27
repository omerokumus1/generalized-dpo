import tiktoken
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel

from args import Args
from enum import Enum


class LLM(Enum):
    unsloth_llama_3_8b_bnb_4bit = "unsloth/llama-3-8b-bnb-4bit"
    unsloth_llama_3_1b_bnb_4bit = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"


def load_llm(llm_name: LLM):
    if llm_name.value.startswith("unsloth/"):
        max_seq_length = Args.max_seq_length  # Choose any! We auto support RoPE Scaling internally!
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_name.value,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        return model, tokenizer
