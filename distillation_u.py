import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, HfArgumentParser
from transformers import DataCollatorForLanguageModeling
import wandb
import datetime

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    teacher_model_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the teacher model name"})
    student_model_name: Optional[str] = field(default="gpt2", metadata={"help": "the student model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "# steps to save the model"})
    log_steps: Optional[int] = field(default=10, metadata={"help": "# steps to log the model"})
    output_dir: Optional[str] = field(default="", metadata={"help": "n steps to save the model"})
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb project name"})
    comments: Optional[str] = field(default="", metadata={"help": "wandb project name"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

teacher_model_name = script_args.teacher_model_name
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_model.eval()

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
tokenizer.pad_token = tokenizer.eos_token

student_model_name = script_args.student_model_name
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

train_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets, 
    batch_size=8,  # Adjust batch size based on your GPU memory
    shuffle=True,
    collate_fn=data_collator
)

# Define optimizer for the student model
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

# Knowledge distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature, alpha):
    # KL Divergence loss for distillation
    loss_kl = nn.KLDivLoss()(F.log_softmax(student_logits / temperature, dim=-1),
                             F.softmax(teacher_logits / temperature, dim=-1)) * (temperature ** 2)
    return alpha * loss_kl

# Training loop with distillation
temperature = 2.0
alpha = 0.9
num_epochs = 3
steps_per_epoch = len(train_dataloader)
os.makedirs(script_args.output_dir, exist_ok=True)
with open(f'{script_args.output_dir}/metadata.txt','w') as file:
    file.writelines(f"""
                    script: distillation_u.py
                    time: {datetime.datetime.now()}
                    teacher_model_name: {script_args.teacher_model_name}
                    student_model_name: {script_args.student_model_name}
                    dataset_name: {script_args.dataset_name}
                    mini_batch_size: {script_args.mini_batch_size}
                    batch_size: {script_args.batch_size}
                    save_steps: {script_args.save_steps}
                    log_steps: {script_args.log_steps}
                    output_dir: {script_args.output_dir}
                    wandb_name: {script_args.wandb_name}
                    comments: {script_args.comments}
                    """)
wandb.init(project="multi-step_distill", name=script_args.wandb_name)

for epoch in range(num_epochs):
    print(epoch)
    student_model.train()
    running_loss = 0.0

    step = 0
    for batch in train_dataloader:
        inputs = batch['input_ids'].to(student_model.device)
        attention_mask = batch['attention_mask'].to(student_model.device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        student_outputs = student_model(input_ids=inputs, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits, temperature, alpha)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % script_args.save_steps == 0:
            save_dir = f"{script_args.output_dir}/checkpoint-{epoch}-{step}"
            os.makedirs(save_dir, exist_ok=True)
            student_model.save_pretrained(save_dir)
        print(f"epoch {epoch}/{num_epochs}, step {step}/{steps_per_epoch}, loss={loss.item()}")
        wandb.log({"distillation_loss": loss.item(),"step": epoch*steps_per_epoch+step})
        step += 1

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

print("Distillation training completed.")