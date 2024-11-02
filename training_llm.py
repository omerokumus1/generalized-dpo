import time

import torch

from args import Args
from dpo_loss import compute_dpo_loss_batch, evaluate_dpo_loss_loader
from prepare_dataset import format_input
from utils import generate_and_print_sample
from torch.optim import Optimizer


def train_model_dpo_simple(
        policy_model, reference_model, train_loader, val_loader,
        optimizer: Optimizer, num_epochs, beta,
        eval_freq, eval_iter, start_context, tokenizer
):
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": []
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()  # Set model to training mode
        loss = None
        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta
            )
            loss = torch.tensor(loss.item(), device=loss.device, requires_grad=True).mean()
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)
                train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                    f"Train reward margins {train_reward_margin:.3f}, "
                    f"Val reward margins {val_reward_margin:.3f}"
                )
                if global_step >= 300:
                    break

        # Print a sample text after each epoch
        generate_and_print_sample(
            model=policy_model,
            tokenizer=tokenizer,
            device=loss.device,
            start_context=start_context
        )

    return tracking


def start_training(policy_model, reference_model, train_loader, val_loader, val_data, tokenizer):
    start_time = time.time()

    torch.manual_seed(Args.torch_seed)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=Args.num_epochs,
        beta=0.1,  # value between 0.1 and 0.5
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[2]),
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return tracking
