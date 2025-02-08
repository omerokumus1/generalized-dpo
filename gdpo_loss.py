from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

import utils
from custom_types import ProcessedBatch
# import ipdb
from loss_commons import compute_dpo_loss, compute_logprobs
import ipdb

def get_max_of_rejected_logprobs(model, batch):
    """
        len(batch["rejecteds"]) = batch_size
        len(rejected_log_probas_list) = batch_size
        len(max_item_indices) = batch_size
    """
    # Put each tensor to the model and compute the log probs
    with torch.no_grad():
        rejected_log_probas_list: List[Tensor] = []
        for i in range(len(batch["rejecteds"])):
            rejected_log_probas_list.append(
                compute_logprobs(
                    logits=model(batch["rejecteds"][i]).logits,
                    labels=batch["rejecteds"][i],
                    selection_mask=batch["rejecteds_mask"][i]
                )
            )
        # Find the index of the maximum value in each tensor
        max_indices = [torch.argmax(tensor).item() for tensor in rejected_log_probas_list]
        tensors_with_max_logprobs = []
        labels_with_max_logprobs = []
        for i,k in enumerate(max_indices):
            tensors_with_max_logprobs.append(batch["rejecteds"][i][k])
            labels_with_max_logprobs.append(batch["rejecteds_mask"][i][k])

        tensors_with_max_logprobs = torch.stack(tensors_with_max_logprobs)
        labels_with_max_logprobs = torch.stack(labels_with_max_logprobs)
    # Compute the log probs of the max rejected log probs
    max_rejected_log_probas = compute_logprobs(
        logits=model(tensors_with_max_logprobs).logits,
        labels=tensors_with_max_logprobs,
        selection_mask=labels_with_max_logprobs
    )

    return max_rejected_log_probas


def get_log_probs(model, batch, is_policy_model: bool):
    """Compute the log probabilities of the chosen and rejected responses for a batch"""
    chosen_log_probas = None
    rejected_log_probas = None
    if is_policy_model:
        # print("get_log_probs is_policy_model")
        chosen_log_probas = compute_logprobs(
            logits=model(batch["chosen"]).logits,
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )

        rejected_log_probas = get_max_of_rejected_logprobs(
            model=model,
            batch=batch
        )
    else:
        with torch.no_grad():
            chosen_log_probas = compute_logprobs(
                logits=model(batch["chosen"]).logits,
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )

            rejected_log_probas = get_max_of_rejected_logprobs(
                model=model,
                batch=batch
            )

    return chosen_log_probas, rejected_log_probas


def compute_gdpo_loss_batch(batch: ProcessedBatch, policy_model, reference_model, beta):
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


def compute_gdpo_loss_loader(data_loader, policy_model, reference_model, beta, num_batches=None):
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
            loss, chosen_rewards, rejected_rewards = compute_gdpo_loss_batch(
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


def evaluate_gdpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """Compute the DPO loss for the training and validation dataset"""

    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_gdpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_gdpo_loss_loader(
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
    chosen_logits = policy_model(batch["chosen"]).logits  # Shape: (2, x, 128256)
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
