import torch
from unsloth import FastLanguageModel

from args import Args
from supported_llms import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_llm(llm: LLM):
    model = None
    tokenizer = None
    if llm.value.startswith("unsloth/"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm.value,
            max_seq_length=Args.max_seq_length,  # Choose any! We auto support RoPE Scaling internally!
            dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            load_in_4bit=True,  # Use 4bit quantization to reduce memory usage. Can be False.
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
        model.to(Args.device)

    elif llm == LLM.ytu_turkish_llama_8b_v01:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            load_in_8bit_fp32_cpu_offload=True,
            device_map='auto'
        )

        tokenizer = AutoTokenizer.from_pretrained(llm.value)
        model = AutoModelForCausalLM.from_pretrained(
            llm.value,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(llm.value)
        model = AutoModelForCausalLM.from_pretrained(llm.value)
        model.to(Args.device)

    return model, tokenizer
