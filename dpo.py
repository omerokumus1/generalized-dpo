import pprint
import time
import json

import torch
from torch.utils.data import DataLoader

import utils
from PreferenceDataset import DpoPreferenceDataset
from args import Args
from dpo_batch_processing import get_dpo_customized_collate_fn
from dpo_loss import evaluate_dpo_loss_loader
from evaluating import print_model_responses
from load_llm import load_llm
from prepare_dataset import read_data, format_input, get_sub_data, get_train_test_validation_data, print_data_lengths
from supported_llms import LLM
from test_util import test_model, test_data_loader
from training_llm import start_training
from utils import write_to_json

Args.method = "dpo"
Args.data_file_path = "data/dpo_data_en.json"
Args.LLM = LLM.unsloth_llama3_8b_instruct
Args.DEBUG = False
Args.use_sub_data = False
Args.is_model_local = False

# * Check Libraries
print("Check Libraries")
utils.library_versions()

# * Ch2. Preparing a preference dataset for DPO
print("\n\nCh2. Preparing a preference dataset for DPO")

'''
Dataset format
[
    {
        "instruction": "Aşağıda bir soru ve bu soru için doğru olabilecek seçenekler A), B), C) şeklinde verilmiştir. Seçenek formatı <HARF><SAĞ PARANTEZ><AÇIKLAMA> şeklindedir. Verilen soruya göre doğru cevabı seç ve açıkla.",
        "input": "Aşağıdaki cümlelerin hangisinde –de ekinin yazımında bir yanlışlık yapılmıştır?\n\nA) Sokakta kimsecikler yoktu.\nB) Caddede oturanlarda bu durumdan habersizdi.\nC) Okulun bahçesinde birkaç öğrenci kalmıştı.\nD) Sizleri de aramızda görmekten mutluluk duyarız.\n",
        "rejected":"A) Sokakta kimsecikler yoktu.",
        "chosen": "Türkçe'de '-de' ekinin ayrı yazılması gerektiği durumlar vardır. 'B' seçeneğinde 'oturanlarda' kelimesi yanlış yazılmıştır; doğru yazımı 'oturanlar da' şeklindedir, çünkü '-de' eki burada ayrı yazılmalıdır.\nDoğru cevap B seçeneğidir."
    },
    .
    .
    .,
]
'''

# ? 2.1. Loading a preference dataset
print("\n\n# 2.1. Loading a preference dataset")
data = read_data(Args.data_file_path)
print("\tNumber of entries:", len(data))

if Args.use_sub_data:
    data = get_sub_data(data)
    print("\tNumber of entries in subdata:", len(data))

print("\n-> Print Data Test ")
pprint.pp(data[50])

print("\n-> Format Input Test ")
model_input = format_input(data[50])
print(model_input)

print("\n-> Desired Response by Entry Test ")
desired_response = f"### Response:\n{data[50]['chosen']}"
print(desired_response)

print("\n-> Possible Responses by Entry Test ")
possible_responses = f"### Response:\n{data[50]['rejected']}"
print(possible_responses)

# ? 2.2. Creating training, validation, and test splits
print("\n\n# 2.2. Creating training, validation, and test splits")
train_data, test_data, val_data = get_train_test_validation_data(data, Args.train_percent, Args.test_percent)

print("\n-> Data Lengths")
print_data_lengths(data, train_data, test_data, val_data)

# ? 2.3. Developing a PreferenceDataset class and batch processing function
print("\n\n# 2.3. Developing a PreferenceDataset class and batch processing function")
print("\tDevice:", Args.device)
customized_collate_fn = get_dpo_customized_collate_fn()
print("\tCustomized Collate Function:")
print(customized_collate_fn)

# * Ch3. Loading a Finetuned LLM for DPO Alignment
print("\n\nCh3. Loading a Finetuned LLM for DPO Alignment")
model, tokenizer = load_llm(Args.LLM)
Args.pad_token_id = tokenizer.eos_token_id

print("\n-> Model Test")
model.eval()
if Args.DEBUG:
    test_model(model, tokenizer)

print("\n -> Loading policy_model and reference_model..")
policy_model = model
reference_model, _ = load_llm(Args.LLM)
reference_model.eval()

# ? 2.4. Creating training, validation, and test set data loaders
print("\n\n# 2.4. Creating training, validation, and test set data loaders")
torch.manual_seed(Args.torch_seed)
train_dataset = DpoPreferenceDataset(train_data, tokenizer, format_input)
train_loader = DataLoader(
    train_dataset,
    batch_size=Args.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=Args.num_workers
)

val_dataset = DpoPreferenceDataset(val_data, tokenizer, format_input)
val_loader = DataLoader(
    val_dataset,
    batch_size=Args.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=Args.num_workers
)

test_dataset = DpoPreferenceDataset(test_data, tokenizer, format_input)
test_loader = DataLoader(
    test_dataset,
    batch_size=Args.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=Args.num_workers
)

if Args.DEBUG:
    test_data_loader(train_loader, tokenizer, "Train Loader")

# * Ch4. Implementing the DPO Loss Function
print("\n\nCh4. Implementing the DPO Loss Function")

# * Ch5. Initial Responses
print("\n\nCh5. Initial Responses")
print("\t-> Initial losses and rewards:")
start_time = time.time()
res = evaluate_dpo_loss_loader(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    beta=0.1,
    eval_iter=5
)
end_time = time.time()

print("\t\tTraining loss:", res["train_loss"])
print("\t\tValidation loss:", res["val_loss"])
print("\t\tTrain reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("\t\tVal reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

print("\n\tFirst 3 of initial model's responses on validation_data:")
print_model_responses(policy_model, reference_model, val_data, tokenizer)

print("\nEmptying cache...")
torch.cuda.empty_cache()

# * Ch6. DPO Training
print("\nCh6. DPO training")
print("\nStarting DPO training...")
tracking = start_training(policy_model, reference_model, train_loader, val_loader, method="dpo")

# Save the tracking data to a JSON file
write_to_json(tracking, "result/dpo_tracking.json")

# Save the model to the Hugging Face Hub
policy_model.push_to_hub("BIGDaTA-Lab/Llama-3.2-1B-4bit-dpo")

# * Ch7. Evaluating DPO Model
print("\n\nCh7. Evaluating DPO Model")
print("\t-> Plotting DPO Loss:")
epochs_tensor = torch.linspace(0, Args.num_epochs, len(tracking["train_losses"]))
utils.plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    train_loss_label="Train loss",
    val_loss_label="Validation loss",
    title="DPO Losses"
)

print("\t-> Final losses and rewards:")
train_reward_margins = [i - j for i, j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
val_reward_margins = [i - j for i, j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

utils.plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=train_reward_margins,
    val_losses=val_reward_margins,
    train_loss_label="Train reward margin",
    val_loss_label="Validation reward margin",
    title="DPO Reward Margins"
)

utils.plot_gpu_usage(
    tokens_seen=tracking["tokens_seen"],
    reserved_gpu_memory=tracking["reserved_gpu_memory"],
    allocated_gpu_memory=tracking["allocated_gpu_memory"],
    title="DPO GPU Usage (MB)"
)

print("\t-> First 3 of final model's responses on validation_data:")
print_model_responses(policy_model, reference_model, val_data, tokenizer)