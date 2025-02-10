import utils
import json
from args import Args
from prepare_dataset import format_input
from utils import generate, text_to_token_ids, token_ids_to_text, write_to_txt, write_to_json
import FastLanguageModel
import re

def print_model_responses(policy_model, reference_model, data, tokenizer, response_count = 3):
    responses = []
    for entry in data[:response_count]:

        input_text = format_input(entry)

        token_ids = generate(
            model=reference_model,
            idx=text_to_token_ids(input_text, tokenizer).to(Args.device),
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
            idx=utils.text_to_token_ids(input_text, tokenizer).to(Args.device),
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
        print(f"\nCorrect response:\n>> {entry['chosen']}")
        print(f"\nReference model response:\n>> {reference_response_text.strip()}")
        print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
        print("\n-------------------------------------\n")

        correct_response = f"\nCorrect response:\n>> {entry['chosen']}"
        reference_response = f"\nReference model response:\n>> {reference_response_text.strip()}"
        policy_response = f"\nPolicy model response:\n>> {policy_response_text.strip()}"

        response = {
            "input": input_text,
            "correct_response": correct_response,
            "reference_response": reference_response,
            "policy_response": policy_response,
        }
        responses.append(response)

    write_to_json(responses, f'{Args.method.upper()} model_responses.json')


def get_model_response(model, input_text, tokenizer):
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(Args.device),
        max_new_tokens=256,
        context_size=Args.max_context_length,
        eos_id=Args.pad_token_id
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    return response_text

def get_model_responses(gdpo_model, dpo_model, reference_model, data, tokenizer, response_count=3):
    dpo_responses = []
    gdpo_responses = []
    reference_responses = []
    correct_responses = []
    inputs = []

    FastLanguageModel.for_inference(gdpo_model)  # Enable native 2x faster inference
    FastLanguageModel.for_inference(dpo_model)  # Enable native 2x faster inference
    FastLanguageModel.for_inference(reference_model)  # Enable native 2x faster inference

    for i, entry in enumerate(data[:response_count]):
        print(f"Processing entry {i+1}/{response_count}")
        input_text = format_input(entry)
        gdpo_responses.append(get_model_response(gdpo_model, input_text, tokenizer))
        dpo_responses.append(get_model_response(dpo_model, input_text, tokenizer))
        reference_responses.append(get_model_response(reference_model, input_text, tokenizer))
        correct_responses.append(entry['chosen'])
        inputs.append(input_text)

    return gdpo_responses, dpo_responses, reference_responses, correct_responses, inputs
