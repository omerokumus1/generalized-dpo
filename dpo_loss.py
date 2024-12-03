from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

import utils
from custom_types import ProcessedBatch
import ipdb

logits_min = -1e35
logits_max = 1e35
clamp_min = 1e-5
clamp_max = 1e35


def clamp(tensor: Tensor) -> Tensor:
    return torch.clamp(tensor, min=clamp_min, max=clamp_max)


def get_logits(model, input: Tensor) -> Tensor:
    logits = model(input).logits
    return torch.clamp(logits, min=logits_min, max=logits_max)


# This function calculates logarithms, and you need to pass the combined
# scores of rejected answers.
def compute_dpo_loss(
        policy_chosen_logprobs,
        policy_rejected_logprobs,
        reference_chosen_logprobs,
        reference_rejected_logprobs,
        beta=0.1,
):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    # reference_model's logits can contain inf values
    losses = clamp(-F.logsigmoid(beta * logits))

    # Optional values to track progress during training
    chosen_rewards = (policy_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (policy_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_logprobs(logits: Tensor, labels: Tensor, selection_mask: Tensor = None) -> Tensor:
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """

    # Labels are the inputs shifted by one
    labels = labels[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]
    log_probs = clamp(F.log_softmax(logits, dim=-1))

    # Gather the log probabilities for the actual labels
    # Here, torch.gather calculates the cross entropy
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # The selection_mask we use here is to optionally ignore prompt and padding tokens
    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)


def get_max_of_rejected_logprobs(model, batch: ProcessedBatch, is_policy_model: bool):
    """
        len(batch["rejecteds"]) = batch_size
        len(rejected_log_probas_list) = batch_size
        len(max_item_indices) = batch_size
    """

    # Put each tensor to the model and compute the log probs
    rejected_log_probas_list: List[Tensor] = []
    with torch.no_grad():
        for i in range(len(batch["rejecteds"])):
            rejected_log_probas_list.append(
                compute_logprobs(
                    logits=get_logits(model, batch["rejecteds"][i]),
                    labels=batch["rejecteds"][i],
                    selection_mask=batch["rejecteds_mask"][i]
                )
            )

    # Get the max of each rejected response
    return torch.max(torch.stack(rejected_log_probas_list))


def get_log_probs(model, batch: ProcessedBatch, is_policy_model: bool):
    """Compute the log probabilities of the chosen and rejected responses for a batch"""
    chosen_log_probas = None
    rejected_log_probas = None
    if is_policy_model:
        # print("get_log_probs is_policy_model")
        chosen_log_probas = compute_logprobs(
            logits=get_logits(model, batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )

        rejected_log_probas = get_max_of_rejected_logprobs(
            model=model,
            batch=batch,
            is_policy_model=is_policy_model
        )
    else:
        with torch.no_grad():
            chosen_log_probas = compute_logprobs(
                logits=get_logits(model, batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )

            rejected_log_probas = get_max_of_rejected_logprobs(
                model=model,
                batch=batch,
                is_policy_model=is_policy_model
            )

    return chosen_log_probas, rejected_log_probas


def compute_dpo_loss_batch(batch: ProcessedBatch, policy_model, reference_model, beta):
    """Compute the DPO loss on an input batch"""
    policy_chosen_log_probas, policy_rejected_log_probas = get_log_probs(
        model=policy_model,
        batch=batch,
        is_policy_model=True
    )

    # Reference model logrporbs
    ref_chosen_log_probas, ref_rejected_log_probas = get_log_probs(
        model=reference_model,
        batch=batch,
        is_policy_model=False
    )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        policy_chosen_logprobs=policy_chosen_log_probas,
        policy_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )

    return loss, chosen_rewards, rejected_rewards


def compute_dpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
    """Apply compute_dpo_loss_batch to a whole data loader"""

    total_loss, total_chosen_rewards, total_rejected_rewards = 0., 0., 0.
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

        else:
            break

    # calculate average
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    return total_loss, total_chosen_rewards, total_rejected_rewards


def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """Compute the DPO loss for the training and validation dataset"""

    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_dpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards
    }

    policy_model.train()
    return res


def dummy_loss_function(batch, policy_model):
    # Extract chosen and rejected tensors from the batch
    chosen_logits = get_logits(policy_model, batch["chosen"])  # Shape: (2, x, 128256)
    # rejecteds_logits = get_logits(policy_model, batch['rejecteds'][0][:2])  # Shape: (3, x, 128256)
    # rejecteds_logits = chosen_logits.clone()

    # Compute a dummy loss: Mean Squared Error between chosen and rejected
    # We will average the error over the 3 options in rejecteds
    # Use chosen as the target for simplicity

    # Calculate MSE loss between chosen and each rejected option
    mse_loss = F.mse_loss(chosen_logits, chosen_logits / 2, reduction='mean')  # Scalar loss
    print("mse_loss:", mse_loss)
    # Dummy rewards (use random values for chosen and rejected rewards for testing)
    chosen_rewards = torch.rand(chosen_logits.size(0))  # Random rewards for 'chosen', shape: (2,)
    rejected_rewards = torch.rand(chosen_logits.size(0),
                                  chosen_logits.size(1))  # Random rewards for 'rejected', shape: (3, x)

    return mse_loss, chosen_rewards, rejected_rewards
