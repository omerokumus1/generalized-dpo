import pprint

import torch
from torch.utils.data import DataLoader

import utils
from PreferenceDataset import PreferenceDataset
from args import Args
from batch_processing import get_customized_collate_fn
from dpo_loss import evaluate_dpo_loss_loader
from evaluating import print_model_responses
from load_llm import load_llm, LLM
from prepare_dataset import read_data, format_input, get_sub_data, get_train_test_validation_data, print_data_lengths
from test_util import test_model, test_data_loader
from training_llm import start_training


def set_arguments():
    Args.pad_token_id = 128001


set_arguments()

# * Check Libraries
print("Check Libraries")
utils.library_versions()

# * Ch2. Preparing a preference dataset for DPO
print("\n\nCh2. Preparing a preference dataset for DPO")

'''
Dataset format
[
    {
        "instruction": "Aşağıda bir soru ve bu soru için doğru olabilecek şıklar A), B), C) şeklinde verilmiştir. Verilen soruya göre doğru cevabı seç ve açıkla.",
        "input": "Aşağıdaki cümlelerin hangisinde –de ekinin yazımında bir yanlışlık yapılmıştır?\n\nA) Sokakta kimsecikler yoktu.\nB) Caddede oturanlarda bu durumdan habersizdi.\nC) Okulun bahçesinde birkaç öğrenci kalmıştı.\nD) Sizleri de aramızda görmekten mutluluk duyarız.\n",
        "output": "Türkçe'de '-de' ekinin ayrı yazılması gerektiği durumlar vardır. 'B' seçeneğinde 'oturanlarda' kelimesi yanlış yazılmıştır; doğru yazımı 'oturanlar da' şeklindedir, çünkü '-de' eki burada ayrı yazılmalıdır.\nDoğru cevap B seçeneğidir.",
        "rejecteds": [
            "A) Sokakta kimsecikler yoktu.",
            "C) Okulun bahçesinde birkaç öğrenci kalmıştı.",
            "D) Sizleri de aramızda görmekten mutluluk duyarız.",
        ],
        "chosen": "Türkçe'de '-de' ekinin ayrı yazılması gerektiği durumlar vardır. 'B' seçeneğinde 'oturanlarda' kelimesi yanlış yazılmıştır; doğru yazımı 'oturanlar da' şeklindedir, çünkü '-de' eki burada ayrı yazılmalıdır.\nDoğru cevap B seçeneğidir."
    },
    .
    .
    .,
]
'''

# ? 2.1. Loading a preference dataset
print("\n\n# 2.1. Loading a preference dataset")
data = read_data(Args.file_path)
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
possible_responses = f"### Response:\n{data[50]['rejecteds']}"
print(possible_responses)

# ? 2.2. Creating training, validation, and test splits
print("\n\n# 2.2. Creating training, validation, and test splits")
train_data, test_data, val_data = get_train_test_validation_data(data, Args.train_percent, Args.test_percent)

print("\n-> Data Lengths")
print_data_lengths(data, train_data, test_data, val_data)

# ? 2.3. Developing a PreferenceDataset class and batch processing function
print("\n\n# 2.3. Developing a PreferenceDataset class and batch processing function")
print("\tDevice:", Args.device)
customized_collate_fn = get_customized_collate_fn()
print("\tCustomized Collate Function:")
print(customized_collate_fn)

# * Ch3. Loading a Finetuned LLM for DPO Alignment
print("\n\nCh3. Loading a Finetuned LLM for DPO Alignment")
model, tokenizer = load_llm(LLM.unsloth_llama_3_8b_bnb_4bit)

print("\n-> Model Test")
model.eval()
if Args.DEBUG:
    test_model(model, tokenizer)

print("\n -> Loading policy_model and reference_model..")
policy_model = model
reference_model, _ = load_llm(LLM.unsloth_llama_3_8b_bnb_4bit)
reference_model.eval()

# ? 2.4. Creating training, validation, and test set data loaders
print("\n\n# 2.4. Creating training, validation, and test set data loaders")
torch.manual_seed(Args.torch_seed)
train_dataset = PreferenceDataset(train_data, tokenizer, format_input)
train_loader = DataLoader(
    train_dataset,
    batch_size=Args.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=Args.num_workers
)

val_dataset = PreferenceDataset(val_data, tokenizer, format_input)
val_loader = DataLoader(
    val_dataset,
    batch_size=Args.batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=Args.num_workers
)

test_dataset = PreferenceDataset(test_data, tokenizer, format_input)
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

# * Ch5. Training the Model
print("\n\nCh5. Training the Model")
print("\t-> Initial losses and rewards:")
res = evaluate_dpo_loss_loader(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    beta=0.1,
    eval_iter=5
)

print("\t\tTraining loss:", res["train_loss"])
print("\t\tValidation loss:", res["val_loss"])
print("\t\tTrain reward margin:", res["train_chosen_reward"] - res["train_rejected_reward"])
print("\t\tVal reward margin:", res["val_chosen_reward"] - res["val_rejected_reward"])

print("\n\tFirst 3 of initial model's responses on validation_data:")
print_model_responses(policy_model, reference_model, val_data, tokenizer)

print("\nEmptying cache...")
torch.cuda.empty_cache()

print("\nStarting training...")
tracking = start_training(policy_model, reference_model, train_loader, val_loader, val_data, tokenizer)

# * Ch6. Evaluating the Model
print("\n\nCh6. Evaluating the Model")
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
    title="Reward Margins"
)

print("\t-> First 3 of final model's responses on validation_data:")
print_model_responses(policy_model, reference_model, val_data, tokenizer)