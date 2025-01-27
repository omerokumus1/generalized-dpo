from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import utils
from args import Args
from custom_types import ProcessedBatch
# import ipdb
from loss_commons import compute_dpo_loss, compute_logprobs


def get_logits(model: nn.Module, input: Tensor):
    print("Inside get_logits")
    input_gpu = input.to(utils.get_model_device(model))
    print("input_gpu:", input_gpu)
    logits = model(input_gpu).logits
    print("logits:", logits)
    return logits

def get_max_of_rejected_logprobs(model, batch):
    """
        len(batch["rejecteds"]) = batch_size
        len(rejected_log_probas_list) = batch_size
        len(max_item_indices) = batch_size
    """
    print("Inside get_max_of_rejected_logprobs")
    # Put each tensor to the model and compute the log probs
    rejected_log_probas_list: List[Tensor] = []
    with torch.no_grad():
        for i in range(len(batch["rejecteds"])):
            print("i:", i)
            print("batch['rejecteds'][i]:", batch["rejecteds"][i])
            print("before compute_logprobs for rejected")
            rejected_log_probas_list.append(
                compute_logprobs(
                    logits=get_logits(model, batch["rejecteds"][i]),
                    labels=batch["rejecteds"][i],
                    selection_mask=batch["rejecteds_mask"][i]
                )
            )
            print("after compute_logprobs for rejected")
            print("computed rejected logprobs:", rejected_log_probas_list[len(rejected_log_probas_list) - 1])

    # Get the max of each rejected response
    print("torch.max(torch.stack(rejected_log_probas_list)):", torch.max(torch.stack(rejected_log_probas_list)))
    return torch.max(torch.stack(rejected_log_probas_list))


def get_log_probs(model, batch, is_policy_model: bool):
    """Compute the log probabilities of the chosen and rejected responses for a batch"""
    print("Inside get_log_probs")
    chosen_log_probas = None
    rejected_log_probas = None
    if is_policy_model:
        print("get_log_probs is_policy_model")
        print("before compute_logprobs")
        chosen_log_probas = compute_logprobs(
            logits=get_logits(model, batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        print("after compute_logprobs")
        print("chosen_log_probas:", chosen_log_probas)

        print("before get_max_of_rejected_logprobs")
        rejected_log_probas = get_max_of_rejected_logprobs(
            model=model,
            batch=batch
        )
        print("after get_max_of_rejected_logprobs")
        print("rejected_log_probas:", rejected_log_probas)
    else:
        print("get_log_probs is reference_model")
        with torch.no_grad():
            print("before compute_logprobs")
            chosen_log_probas = compute_logprobs(
                logits=get_logits(model, batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )
            print("after compute_logprobs")
            print("chosen_log_probas:", chosen_log_probas)

            print("before get_max_of_rejected_logprobs")
            rejected_log_probas = get_max_of_rejected_logprobs(
                model=model,
                batch=batch
            )
            print("after get_max_of_rejected_logprobs")
            print("rejected_log_probas:", rejected_log_probas)

    return chosen_log_probas, rejected_log_probas


def put_input_back_to_device(batch):
    batch["chosen"].to(Args.data_device)
    for i in range(len(batch["rejecteds"])):
        batch["rejecteds"][i].to(Args.data_device)

def compute_gdpo_loss_batch(batch: ProcessedBatch, policy_model, reference_model, beta):
    """Compute the DPO loss on an input batch"""
    print("Inside compute_gdpo_loss_batch")
    print("before get_log_probs for policy_model")
    policy_chosen_log_probas, policy_rejected_log_probas = get_log_probs(
        model=policy_model,
        batch=batch,
        is_policy_model=True
    )
    print("After get_log_probs for policy_model")
    print("policy_chosen_log_probas:", policy_chosen_log_probas)
    print("policy_rejected_log_probas:", policy_rejected_log_probas)

    # Reference model logrporbs
    print("before get_log_probs for reference_model")
    ref_chosen_log_probas, ref_rejected_log_probas = get_log_probs(
        model=reference_model,
        batch=batch,
        is_policy_model=False
    )
    print("After get_log_probs for reference_model")
    print("ref_chosen_log_probas:", ref_chosen_log_probas)
    print("ref_rejected_log_probas:", ref_rejected_log_probas)

    # Compute the DPO loss
    print("before compute_dpo_loss")
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        policy_chosen_logprobs=policy_chosen_log_probas,
        policy_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    print("After compute_dpo_loss")
    print("loss:", loss)
    print("chosen_rewards:", chosen_rewards)
    print("rejected_rewards:", rejected_rewards)

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
            print("batch:", i)
            print("Before compute_gdpo_loss_batch")
            loss, chosen_rewards, rejected_rewards = compute_gdpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            print("After compute_gdpo_loss_batch")
            print("loss:", loss.item())
            print("chosen_rewards:", chosen_rewards.item())
            print("rejected_rewards:", rejected_rewards.item())
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()

            print("total_loss:", total_loss)
            print("total_chosen_rewards:", total_chosen_rewards)
            print("total_rejected_rewards:", total_rejected_rewards)
        else:
            break

    # calculate average
    total_loss /= num_batches
    total_chosen_rewards /= num_batches
    total_rejected_rewards /= num_batches
    print("Avg total_loss:", total_loss)
    print("Avg total_chosen_rewards:", total_chosen_rewards)
    print("Avg total_rejected_rewards:", total_rejected_rewards)
    return total_loss, total_chosen_rewards, total_rejected_rewards


def evaluate_gdpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """Compute the DPO loss for the training and validation dataset"""

    policy_model.eval()
    with torch.no_grad():
        print("before compute_gdpo_loss_loader for train_loader")
        train_loss, train_chosen_rewards, train_rejected_rewards = compute_gdpo_loss_loader(
            data_loader=train_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )
        print("After compute_gdpo_loss_loader for train_loader")
        print("train_loss:", train_loss)
        print("train_chosen_rewards:", train_chosen_rewards)
        print("train_rejected_rewards:", train_rejected_rewards)

        print("before compute_gdpo_loss_loader for val_loader")
        val_loss, val_chosen_rewards, val_rejected_rewards = compute_gdpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter
        )
        print("After compute_gdpo_loss_loader for val_loader")
        print("val_loss:", val_loss)
        print("val_chosen_rewards:", val_chosen_rewards)
        print("val_rejected_rewards:", val_rejected_rewards)

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
