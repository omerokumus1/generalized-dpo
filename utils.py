# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-6.
# This file can be run as a standalone script.


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken
from tiktoken import Encoding
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version

from args import Args
from custom_types import ProcessedBatch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond).logits

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[0][:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = Args.max_context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text)  # Compact print format
    model.train()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist(), skip_special_tokens=True)


def decode_tokens_from_batch(token_ids, tokenizer) -> str:
    ids_in_python_list = token_ids.flatten().tolist()
    return tokenizer.decode(ids_in_python_list, skip_special_tokens=True)


def extract_response(response_text: str, input_text: str) -> str:
    return response_text[len(input_text):].replace("### Response:", "").strip()


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, train_loss_label, val_loss_label, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label=train_loss_label)
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label=val_loss_label)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.legend(loc="best")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{Args.method.upper()} loss-plot.pdf")
    plt.show()


def plot_gpu_usage(tokens_seen, reserved_gpu_memory, allocated_gpu_memory, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot reserved and allocated GPU memory against tokens seen
    ax1.plot(tokens_seen, reserved_gpu_memory, label="Reserved GPU memory")
    ax2.plot(tokens_seen, allocated_gpu_memory, linestyle="-.", label="Allocated GPU memory")
    ax1.set_xlabel("Tokens seen")
    ax2.set_xlabel("Tokens seen")
    ax1.set_ylabel("Memory (MB)")
    ax2.set_ylabel("Memory (MB)")
    ax1.set_title(title)
    ax2.set_title(title)
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{Args.method.upper()} gpu-usage-plot.pdf")
    plt.show()


def library_versions():
    pkgs = [
        "tiktoken",  # Tokenizer
        "torch",  # Deep learning library
        "tqdm",  # Progress bar
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")


def processed_batch_eq(self: ProcessedBatch, other: ProcessedBatch):
    prompt_eq = all([torch.equal(self["prompt"][i], other["prompt"][i]) for i in range(len(self["prompt"]))])
    chosen_eq = all([torch.equal(self["chosen"][i], other["chosen"][i]) for i in range(len(self["chosen"]))])
    rejecteds_eq = all([all([torch.equal(self["rejecteds"][i][j], other["rejecteds"][i][j]) for j in
                             range(len(self["rejecteds"][i]))]) for i in range(len(self["rejecteds"]))])
    rejecteds_mask_eq = all([all([torch.equal(self["rejecteds_mask"][i][j], other["rejecteds_mask"][i][j]) for j in
                                  range(len(self["rejecteds_mask"][i]))]) for i in
                             range(len(self["rejecteds_mask"]))])
    chosen_mask_eq = all(
        [torch.equal(self["chosen_mask"][i], other["chosen_mask"][i]) for i in range(len(self["chosen_mask"]))])

    if not prompt_eq:
        print("Prompt not equal")
        print("self['prompt']:", self["prompt"])
        print('other["prompt"]:', other["prompt"])
    if not chosen_eq:
        print("Chosen not equal")
        print('self["chosen"]:', self["chosen"])
        print('other["chosen"]:', other["chosen"])
    if not rejecteds_eq:
        print("Rejecteds not equal")
        print('self["rejecteds"]:', self["rejecteds"])
        print('other["rejecteds"]:', other["rejecteds"])
    if not rejecteds_mask_eq:
        print("Rejecteds Mask not equal")
        print('self["rejecteds_mask"]:', self["rejecteds_mask"])
        print('other["rejecteds_mask"]:', other["rejecteds_mask"])

    return prompt_eq and chosen_eq and rejecteds_eq and rejecteds_mask_eq and chosen_mask_eq


def processed_batch_clone(self: ProcessedBatch):
    return {
        "prompt": [t.clone() for t in self["prompt"]],
        "chosen": [t.clone() for t in self["chosen"]],
        "rejecteds": [[t.clone() for t in l] for l in self["rejecteds"]],
        "rejecteds_mask": [[t.clone() for t in l] for l in self["rejecteds_mask"]],
        "chosen_mask": [t.clone() for t in self["chosen_mask"]]
    }


def detect_processed_batch_in_place_operations(batch: ProcessedBatch):
    pass


def print_processed_batch_shapes(batch: ProcessedBatch):
    print(
        batch["chosen"].shape,
        [rejected.shape for rejected in batch["rejecteds"]],  # !!! Must print torch.Size(8, <Int>) for each
    )


def print_computation_graph(tensor, visited=None):
    if visited is None:
        visited = set()  # To avoid visiting the same node multiple times

    # Avoid revisiting the same node in the computation graph
    if tensor in visited:
        return
    visited.add(tensor)

    # Print the current tensor details
    print(f"Tensor: {tensor}")
    print(f"grad_fn: {tensor.grad_fn}")
    print("-" * 50)

    # Recursively traverse the graph using the grad_fn's next_functions
    if tensor.grad_fn is not None:
        for next_tensor, _ in tensor.grad_fn.next_functions:
            if next_tensor is not None and isinstance(next_tensor, torch.autograd.Function):
                # Try to get the associated tensor from next_tensor
                # Note: next_tensor doesn't directly store tensors, but we can trace the computation
                try:
                    associated_tensor = next_tensor.variable
                except AttributeError:
                    associated_tensor = None

                # Print if there is an associated tensor
                if associated_tensor is not None:
                    print_computation_graph(associated_tensor, visited)


def check_inf_and_nan(tens):
    inf = False
    nan = False
    if torch.isinf(tens).any().item():
        print("Tensor contains inf values.")
        inf = True
    if torch.isnan(tens).any().item():
        print("Tensor contains inf values.")
        nan = True

    if not inf and not nan:
        print("Tensor does not contain inf or nan values.")
        return False

    return True


def check_inf_and_nan_for_model(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                print(f"Gradient for {name} contains inf or NaN values.")


def debug_forward_pass(model, input):
    with torch.no_grad():
        output = input
        for name, layer in model.named_children():
            output = layer(output).logits
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Layer {name} produced NaN values.")
                break


def monitor_gpu_usage():
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs
    print("\nGPU Memory Usage:")
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}:")
        print(f"\tAllocated Memory: {torch.cuda.memory_allocated(gpu_id) / 1e6:.2f} MB")
        print(f"\tReserved Memory:  {torch.cuda.memory_reserved(gpu_id) / 1e6:.2f} MB")
        print("-" * 40)

    # print("nvidia-smi Output:")
    # os.system("nvidia-smi")
    print("-" * 40)
    print("\n")


def print_peak_gpu_usage():
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs

    print("\nPeak GPU Memory Usage:")
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}:")
        print(f"\tMax Allocated:    {torch.cuda.max_memory_allocated(gpu_id) / 1e6:.2f} MB")
        print(f"\tMax Reserved:     {torch.cuda.max_memory_reserved(gpu_id) / 1e6:.2f} MB")
        print("-" * 40)


def get_model_device(model):
    """
    Returns the device where the model's parameters are located.
    """
    return next(model.parameters()).device
