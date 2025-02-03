import torch
from torch.utils.data import DataLoader

from args import Args
from utils import decode_tokens_from_batch
from gdpo_loss import compute_gdpo_loss_batch
from utils import generate, token_ids_to_text
from torch.utils.data import DataLoader
from tiktoken import Encoding


def test_model(model, tokenizer):
    prompt = """Below is an instruction that describes a task. Write a response
that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(Args.device)
    token_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=35,
        context_size=Args.max_context_length,
        eos_id=tokenizer.vocab_size
    )
    response = token_ids_to_text(token_ids, tokenizer)
    print(response)


def test_data_loader(loader, tokenizer, loader_name: str):
    print("\n\nLoader Test")
    print(f"\t->Loader name: {loader_name}")
    for batch in loader:
        print("****Chosen****")
        print(
            decode_tokens_from_batch(batch["chosen"], tokenizer=tokenizer))

        print("\n****Rejecteds****")
        for rejected in batch["rejecteds"]:
            print(decode_tokens_from_batch(rejected, tokenizer=tokenizer))

        print("\n\n****Next****")
        break

    print(f"\n-> {loader_name} torch shapes:")
    i = 0
    for batch in loader:
        print(
            batch["chosen"].shape,
            [rejected.shape for rejected in batch["rejecteds"]],  # !!! Must print torch.Size(8, <Int>) for each
        )
        i += 1
        if i == 5:
            break


def test_compute_dpo_loss_batch(batch, policy_model, reference_model):
    print("\n\nCompute DPO Loss Batch Test")
    with torch.no_grad():
        loss = compute_gdpo_loss_batch(batch, policy_model, reference_model, beta=0.1)
    print(loss)



def test_decode_tokens_from_batch(data_loader: DataLoader, tokenizer: Encoding):
    for batch in data_loader:
        print("batch.keys:", batch.keys())
        print("batch['prompt']:", batch["prompt"])
        print("First propt's token ids")
        print(batch["prompt"][0])
        first_prompt_decoded = decode_tokens_from_batch(
            token_ids=batch["prompt"][0],  # [0] for the first entry in the batch
            tokenizer=tokenizer,
        )
        print("\nFirst prompt decoded from token ids")
        print(first_prompt_decoded)
        break
