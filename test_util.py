import torch
import torch.nn.functional as F
from tiktoken import Encoding
from torch.utils.data import DataLoader

from args import Args
from utils import decode_tokens_from_batch
from utils import generate, token_ids_to_text


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


def print_raw_loss_values(losses: torch.Tensor):
    """
    Prints raw loss values and ensures they are positive.

    Args:
        losses: Tensor of loss values to inspect.
    """
    # Convert losses to numpy for better printing
    losses_np = losses.detach().cpu().numpy()

    # Print the raw loss values
    # print(f"Raw Loss Values (first 10): {losses_np[:10]}")

    # Check if all losses are positive
    if (losses_np < 0).any():
        print("Warning: Some loss values are negative!")
    # else:
    #     print("All loss values are positive!")

    # Optionally, print summary statistics
    # print(f"Loss Values Stats: Mean={losses.mean().item()}, Min={losses.min().item()}, Max={losses.max().item()}")


def print_reward_margins(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor):
    """
    Prints reward margins and ensures chosen rewards are higher than rejected ones.

    Args:
        chosen_rewards: Tensor of chosen rewards.
        rejected_rewards: Tensor of rejected rewards.
    """
    # Convert rewards to numpy for better printing
    chosen_rewards_np = chosen_rewards.detach().cpu().numpy()
    rejected_rewards_np = rejected_rewards.detach().cpu().numpy()

    # Print the first 10 rewards for inspection
    # print(f"Chosen Rewards (first 10): {chosen_rewards_np[:10]}")
    # print(f"Rejected Rewards (first 10): {rejected_rewards_np[:10]}")

    # Check if chosen rewards are higher than rejected rewards
    if (chosen_rewards_np < rejected_rewards_np).any():
        print("Warning: Some chosen rewards are lower than rejected rewards!")
    # else:
    #     print("All chosen rewards are higher than rejected rewards!")

    # Optionally, print summary statistics
    # print(
    #     f"Chosen Rewards Stats: Mean={chosen_rewards.mean().item()}, Min={chosen_rewards.min().item()}, Max={chosen_rewards.max().item()}")
    # print(
    #     f"Rejected Rewards Stats: Mean={rejected_rewards.mean().item()}, Min={rejected_rewards.min().item()}, Max={rejected_rewards.max().item()}")


def print_log_prob_range(log_probs: torch.Tensor):
    """
    Prints the range of log probabilities and ensures values are valid (negative and reasonable).

    Args:
        log_probs: Tensor of log probabilities.
    """
    # Convert log probabilities to numpy for better printing
    log_probs_np = log_probs.detach().cpu().numpy()

    # Print the first 10 log probabilities for inspection
    # print(f"Log Probabilities (first 10): {log_probs_np[:10]}")

    # Check if all log probabilities are negative
    if (log_probs_np >= 0).any():
        print("Warning: Some log probabilities are non-negative!")
    # else:
    #     print("All log probabilities are negative (as expected).")

    # Check if log probabilities are within a reasonable range (e.g., -100 to 0)
    if (log_probs_np < -100).any():
        print("Warning: Some log probabilities are too low (less than -100)!")

    if torch.isnan(log_probs).any():
        print("Warning: Some log probabilities contain NaN values!")

    # Optionally, print summary statistics
    # print(f"Log Probability Stats: Mean={log_probs.mean().item()}, Min={log_probs.min().item()}, Max={log_probs.max().item()}")


def check_gradient_flow(model):
    """
    Check the gradient flow of a model to ensure that gradients are not all zero.

    Args:
        model: The model whose gradients will be checked.
    """
    # Iterate through all parameters in the model
    for name, param in model.named_parameters():
        # Check if the parameter requires gradients
        if param.requires_grad:
            # Check if gradients are not zero
            if param.grad is None:
                print(f"Warning: Parameter {name} has no gradient!")
            elif torch.all(param.grad == 0):
                print(f"Warning: Parameter {name} has zero gradient!")
            # else:
            #     print(f"Parameter {name}: Gradient is non-zero.")


def compare_with_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None):
    """
    Compare the computed loss with cross-entropy loss and ensure the values are reasonable.

    Args:
        logits: Tensor of logits (predictions) of shape (batch_size, num_tokens, vocab_size).
        labels: Tensor of true labels of shape (batch_size, num_tokens).
        mask: Optional tensor for shape (batch_size, num_tokens) to ignore padding tokens.
    """
    # Compute Cross-Entropy Loss using PyTorch's built-in function (without any mask applied)
    logits = logits.view(-1, logits.size(-1))  # Reshape to (batch_size * num_tokens, vocab_size)
    labels = labels.view(-1)  # Flatten to (batch_size * num_tokens,)
    ce_loss = F.cross_entropy(logits, labels, reduction='none')

    # If mask is provided, apply the mask to ignore padding tokens
    if mask is not None:
        mask = mask.view(-1)
        ce_loss = ce_loss * mask
        ce_loss = ce_loss.sum() / mask.sum()  # Average the loss, ignoring the padding tokens
    else:
        ce_loss = ce_loss.mean()  # Average over all tokens

    # Compute the comparison loss
    # Assuming you are using the logits directly in your DPO loss or another loss function
    computed_loss = logits.mean()  # Replace this with your computed loss if necessary

    # Print the cross-entropy loss for comparison
    # print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")
    # print(f"Computed Loss: {computed_loss.item():.4f}")

    # Ensure computed loss is within a reasonable range
    if computed_loss > 10 * ce_loss:
        print("Warning: Computed loss is much higher than the Cross-Entropy loss.")
    elif computed_loss < 0.1 * ce_loss:
        print("Warning: Computed loss is much lower than the Cross-Entropy loss.")
    # else:
    #     print("Computed loss seems reasonable compared to Cross-Entropy loss.")

# Example usage:
# Assuming `logits`, `labels`, and optionally `mask` are available from your model
# compare_with_cross_entropy_loss(logits, labels, mask)
