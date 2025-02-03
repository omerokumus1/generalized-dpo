import torch

from custom_types import ProcessedBatch, DpoProcessedBatch
from loss_commons import compute_dpo_loss, compute_logprobs


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

        rejected_log_probas = compute_logprobs(
            logits=model(batch["rejected"]).logits,
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )
    else:
        with torch.no_grad():
            chosen_log_probas = compute_logprobs(
                logits=model(batch["chosen"]).logits,
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"]
            )

            rejected_log_probas = compute_logprobs(
                logits=model(batch["rejected"]).logits,
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"]
            )

    return chosen_log_probas, rejected_log_probas


def compute_dpo_loss_batch(batch: DpoProcessedBatch, policy_model, reference_model, beta):
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
