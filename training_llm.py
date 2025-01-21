import time
import torch

import utils
from args import Args
from custom_types import ProcessedBatch
from gdpo_loss import compute_gdpo_loss_batch, evaluate_gdpo_loss_loader, dummy_loss_function
from prepare_dataset import format_input
from torch.optim import Optimizer
import traceback


def train_model_gdpo(
        policy_model, reference_model, train_loader, val_loader,
        optimizer: Optimizer, num_epochs, beta,
        eval_freq, eval_iter
):
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": [],
        "reserved_gpu_memory": [],
        "allocated_gpu_memory": [],
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    try:
        for epoch in range(num_epochs):
            policy_model.train()  # Set model to training mode
            loss = None
            for batch_idx, batch in enumerate(train_loader):
                batch: ProcessedBatch = batch
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

                loss, chosen_rewards, rejected_rewards = compute_gdpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    beta=beta
                )
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients

                tokens_seen += batch["chosen"].numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    res = evaluate_gdpo_loss_loader(
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
                    tracking["reserved_gpu_memory"].append(f"{torch.cuda.memory_reserved() / 1e6:.2f}")
                    tracking["allocated_gpu_memory"].append(f"{torch.cuda.memory_allocated() / 1e6:.2f}")
                    train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                    val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                    print()
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Val reward margins {val_reward_margin:.3f}, "
                    )
                    utils.monitor_gpu_usage()

    except:
        print("********** Exception Occurred **********")
        traceback.print_exc()
        utils.print_nvidia_smi()

    utils.print_peak_gpu_usage()

    return tracking


def train_model_dpo(
        policy_model, reference_model, train_loader, val_loader,
        optimizer: Optimizer, num_epochs, beta,
        eval_freq, eval_iter
):
    # Initialize lists to track losses and tokens seen
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": [],
        "reserved_gpu_memory": [],
        "allocated_gpu_memory": [],
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    try:
        for epoch in range(num_epochs):
            policy_model.train()  # Set model to training mode
            loss = None
            for batch_idx, batch in enumerate(train_loader):
                batch: ProcessedBatch = batch
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

                loss, chosen_rewards, rejected_rewards = compute_gdpo_loss_batch(
                    batch=batch,
                    policy_model=policy_model,
                    reference_model=reference_model,
                    beta=beta
                )
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients

                tokens_seen += batch["chosen"].numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    res = evaluate_gdpo_loss_loader(
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
                    tracking["reserved_gpu_memory"].append(f"{torch.cuda.memory_reserved() / 1e6:.2f}")
                    tracking["allocated_gpu_memory"].append(f"{torch.cuda.memory_allocated() / 1e6:.2f}")
                    train_reward_margin = res["train_chosen_reward"] - res["train_rejected_reward"]
                    val_reward_margin = res["val_chosen_reward"] - res["val_rejected_reward"]

                    print()
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                        f"Train reward margins {train_reward_margin:.3f}, "
                        f"Val reward margins {val_reward_margin:.3f}, "
                    )
                    utils.monitor_gpu_usage()

    except:
        traceback.print_exc()

    utils.print_peak_gpu_usage()

    return tracking


def start_training(policy_model, reference_model, train_loader, val_loader, method="gdpo"):
    start_time = time.time()

    torch.manual_seed(Args.torch_seed)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=5e-5, weight_decay=0.01)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        if method == "gdpo":
            tracking = train_model_gdpo(
                policy_model=policy_model,
                reference_model=reference_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=Args.num_epochs,
                beta=0.1,  # value between 0.1 and 0.5
                eval_freq=5,
                eval_iter=5
            )
            policy_model.save_pretrained(f"{Args.model_path_prefix}/gdpo/model")

        elif method == "dpo":
            tracking = train_model_dpo(
                policy_model=policy_model,
                reference_model=reference_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=Args.num_epochs,
                beta=0.1,  # value between 0.1 and 0.5
                eval_freq=5,
                eval_iter=5,
            )
            policy_model.save_pretrained(f"{Args.model_path_prefix}/dpo/model")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    return tracking
