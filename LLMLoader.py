import torch
from tiktoken import Encoding

from args import Args
from utils import GPTModel, generate, text_to_token_ids, token_ids_to_text


class LLMLoader:
    @staticmethod
    def load_and_eval_model() -> GPTModel:
        print("\n-> Arranging Model Configurations...")
        Args.BASE_CONFIG.update(Args.model_configs[Args.CHOOSE_MODEL])

        print("\n-> Loading Fine-tuned Model..")
        model = GPTModel(Args.BASE_CONFIG)
        model.load_state_dict(
            torch.load(
                Args.finetuned_model_path,
                map_location=torch.device("cpu"),
                weights_only=True
            )
        )
        model.eval()
        return model

    @staticmethod
    def test_model(prompt: str, tokenizer: Encoding, model: GPTModel):
        torch.manual_seed(123)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer),
            max_new_tokens=35,
            context_size=Args.BASE_CONFIG["context_length"],
            eos_id=50256
        )

        print(token_ids_to_text(token_ids, tokenizer))
