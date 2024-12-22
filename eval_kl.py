import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser
from datasets import load_dataset
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional
import time
import wandb
import os

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    teacher_model_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the teacher model name"})
    student_model_name: Optional[str] = field(default="gpt2", metadata={"help": "the student model name"})
    tokenizer_name: Optional[str] = field(default="gpt2", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    save_steps: Optional[int] = field(default=10, metadata={"help": "# steps to save the model"})
    log_steps: Optional[int] = field(default=10, metadata={"help": "# steps to log the model"})
    num_steps: Optional[int] = field(default=500, metadata={"help": "# total finetuning steps"})
    save_best: Optional[int] = field(default=0, metadata={"help": "0 if save at regular frequency, 1 if only save best checkpoint"})
    output_dir: Optional[str] = field(default="", metadata={"help": "n steps to save the model"})
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb project name"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

start_time = time.perf_counter()

# Load pretrained model and tokenizer
teacher_model_name = script_args.teacher_model_name #"facebook/opt-350m"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_model.eval()

student_model_name = script_args.student_model_name #"facebook/opt-125m"
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
print(student_model_name)
print(student_model.transformer.h)

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

dataloader = torch.utils.data.DataLoader(
    tokenized_datasets, 
    batch_size=4,  # Adjust batch size based on your GPU memory
    shuffle=True,
    collate_fn=data_collator
)

# Knowledge distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature, alpha):
    # KL Divergence loss for distillation
    loss_kl = torch.nn.KLDivLoss()(F.log_softmax(student_logits / temperature, dim=-1),
                             F.softmax(teacher_logits / temperature, dim=-1)) * (temperature ** 2)
    return alpha * loss_kl

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)

# Training loop
temperature = 2.0
alpha = 0.9
# teacher_model.eval()
student_model.eval()
KL = 0

for step, batch in enumerate(dataloader):
    if step > script_args.num_steps:
        break
    # Move batch to the same device as the model
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward pass
    student_outputs = student_model(input_ids, attention_mask=attention_mask, labels=input_ids)
    student_logits = student_outputs.logits

    teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, labels=input_ids)
    teacher_logits = teacher_outputs.logits

    loss_kl = distillation_loss(student_logits, teacher_logits, temperature, alpha)
    KL += loss_kl.detach().item()

    print(f'step {step}, KL={loss_kl}')
    if step > script_args.num_steps:
        break

print(f"kl:{KL/script_args.num_steps}")