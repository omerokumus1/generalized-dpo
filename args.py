import torch
from supported_llms import LLM


class Args:
    DEBUG = False
    data_file_path: str = "data/gdpo_data_en.json"
    finetuned_model_path = "gpt2-medium355M-sft.pth"
    LLM = LLM.meta_llama3_1b_instruct
    model_path_prefix = "cache/llama3_1b"
    is_model_local = True

    train_percent: float = 0.85
    test_percent: float = 0.1
    sub_data_size: int = 1000
    use_sub_data = False

    pad_token_id = 128001

    max_seq_length: int = 1024
    max_context_length: int = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")

    num_epochs = 1
    batch_size: int = 8
    mask_prompt_tokens: bool = True

    num_workers: int = 0
    torch_seed: int = 123

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
