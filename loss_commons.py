import torch
import torch.nn.functional as F
from torch import Tensor

from test_util import print_raw_loss_values, print_reward_margins, print_log_prob_range, compare_with_cross_entropy_loss

clamp_max, clamp_min = 20, -20


# Stable smoothed logsigmoid loss
# def smooth_loss(logits, beta, smoothing=0.1):
#     """Compute smoothed logsigmoid loss to prevent overconfidence."""
#     loss = -torch.log1p(torch.exp(-beta * logits))
#     return (1 - smoothing) * loss + smoothing * (-torch.log1p(torch.exp(beta * logits)))


def compute_dpo_loss(
        policy_chosen_logprobs: Tensor,
        policy_rejected_logprobs: Tensor,
        reference_chosen_logprobs: Tensor,
        reference_rejected_logprobs: Tensor,
        beta=0.1,
        smoothing=0.1  # Label smoothing for stability
) -> [Tensor, Tensor, Tensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically in range (0.1 to 0.5). Lower values make it more conservative.
        smoothing: Label smoothing factor (default = 0.1).

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    # Compute log-ratios
    policy_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    # policy_logratios = torch.clamp(policy_logratios, clamp_min, clamp_max)

    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    # reference_logratios = torch.clamp(reference_logratios, clamp_min, clamp_max)

    logits = policy_logratios - reference_logratios
    # logits = torch.clamp(logits, clamp_min, clamp_max)

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits)
    # TODO check the range of losses, which should be (-inf, 0] theoretically, [-20, 0] in practice
    # * apply clamping with[-20, 0] if numerical instability occurs
    # print_raw_loss_values(losses)

    # Detach rewards for monitoring
    chosen_rewards: Tensor = (policy_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards: Tensor = (policy_rejected_logprobs - reference_rejected_logprobs).detach()

    # print_reward_margins(chosen_rewards, rejected_rewards)

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

    # * PyTorch handles all the numerical stability and internal calculations,
    # * making it easier and less error-prone.

    # Truncate logits to match the labels num_tokens
    # TODO check if logits contains inf or nan, which should be [-10, 10].
    # * apply clamping with [-20, 20] if numerical instability occurs
    logits = logits[:, :-1, :]

    # Normally, Logit values can be negative, positive, or zero. Softmax will distribute them over [0, 1]
    # (However, normalization above ensures that the largest value is 0)
    # and summing them results in 1
    # log_probs values are negative (over (-inf, 0]) since log function
    log_probs = F.log_softmax(logits, dim=-1)
    # TODO check the range of log_probs, which should be (-inf, 0] theoretically, [-20, 0] in practice
    # * apply clamping with [-20, 0] if numerical instability occurs
    print_log_prob_range(log_probs)

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

        # Calculate the average log probability excluding padding tokens by mask.sum(-1)
        # This averages over the tokens, so the shape is (batch_size, num_tokens)
        # Compute valid tokens count
        # !! If valid_tokens is 0, it means no active token is in there- all tokens are padding or only prompt tokens
        valid_tokens = mask.sum(-1)

        # Avoid division by zero
        # avg_log_prob = torch.where(
        #     valid_tokens > 0,
        #     selected_log_probs.sum(-1) / valid_tokens,
        #     torch.tensor(0.0, device=logits.device)
        # )
        # valid_tokens = valid_tokens.clamp(min=1)  # Avoid division by 0
        avg_log_prob = selected_log_probs.sum(-1) / valid_tokens

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)
