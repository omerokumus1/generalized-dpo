import torch
from torch import Tensor
import torch.nn.functional as F


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
    reference_chosen_logprobs = reference_chosen_logprobs.to(policy_chosen_logprobs.device)
    reference_rejected_logprobs = reference_rejected_logprobs.to(policy_rejected_logprobs.device)
    model_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    reference_logratios = (reference_chosen_logprobs - reference_rejected_logprobs).to(model_logratios.device)
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    # reference_model's logits can contain inf values
    losses = -F.logsigmoid(beta * logits)

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
    labels = labels[:, 1:].clone().to(logits.device)

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    # Here, torch.gather calculates the cross entropy
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # The selection_mask we use here is to optionally ignore prompt and padding tokens
    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone().to(logits.device)

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)
