import utils
from args import Args
from prepare_dataset import format_input
from utils import generate, text_to_token_ids, token_ids_to_text


def print_model_responses(policy_model, reference_model, data, tokenizer, response_count = 3):
    for entry in data[:response_count]:

        input_text = format_input(entry)

        token_ids = generate(
            model=reference_model,
            idx=text_to_token_ids(input_text, tokenizer).to(utils.get_model_device(reference_model)),
            max_new_tokens=256,
            context_size=Args.max_context_length,
            eos_id=Args.pad_token_id
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        reference_response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        token_ids = generate(
            model=policy_model,
            idx=utils.text_to_token_ids(input_text, tokenizer).to(utils.get_model_device(policy_model)),
            max_new_tokens=256,
            context_size=Args.max_context_length,
            eos_id=Args.pad_token_id
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        policy_response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nReference model response:\n>> {reference_response_text.strip()}")
        print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
        print("\n-------------------------------------\n")