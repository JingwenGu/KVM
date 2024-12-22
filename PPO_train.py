env_choice = 'kv'
log = 'trained_checkpoints/log_llama_1212_kv-1.txt'
save_path = 'trained_checkpoints/KV_llama_1212_1'
total_steps = 1500000
lr = 3e-4
ent_coef = 0.01

import datetime
with open(log,'a') as file:
    file.writelines([f'{datetime.datetime.now()}>> Start\n'])

import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset,load_dataset
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from PPOCallback import PPOCallback

with open(log,'a') as file:
    file.writelines([f'{datetime.datetime.now()}>> Finished imports\n'])

# tokenizer = AutoTokenizer.from_pretrained('gpt2')
# language_model = AutoModelForCausalLM.from_pretrained('gpt2')

token = "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"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token
language_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=token, 
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map="auto",
)

max_length = 100
horizon = 1024

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = []
for sentence in dataset['text']:
    ids = tokenizer(sentence, return_tensors="pt").input_ids
    if ids.shape[1] > max_length:
        texts.append(sentence)
new_dataset = Dataset.from_dict({'text': texts})

tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', padding="max_length", truncation=True, max_length=1024)
tokenized_datasets = new_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

dataloader = torch.utils.data.DataLoader(
    tokenized_datasets, 
    batch_size=1,
    shuffle=True,
    collate_fn=data_collator
)

gym.register(id='EnvKV', entry_point='Env_KV:Env_KV')
gym.register(id='EnvID', entry_point='Env_ID:Env_ID')
env_kwargs = {
    'language_model': language_model, 
    'tokenizer': tokenizer, 
    'dataloader': dataloader, 
    'max_length': max_length, 
    'seed': -1, 
    'horizon': horizon,
    'log': log
}

if env_choice == 'kv':
    kv_vec_env_1 = make_vec_env('EnvKV', n_envs=1, env_kwargs=env_kwargs)
    kv_vec_env_train = make_vec_env('EnvKV', n_envs=1, env_kwargs=env_kwargs)
elif env_choice == 'id':
    kv_vec_env_1 = make_vec_env('EnvID', n_envs=1, env_kwargs=env_kwargs)
    kv_vec_env_train = make_vec_env('EnvID', n_envs=1, env_kwargs=env_kwargs)

hyperparameters = {
    "n_steps": 1000,
    "policy_kwargs": {
        "net_arch": {
            "pi": [128, 128, 128],
            "vf": [256, 256, 256],
            #"activation_fn": "tanh",
            "activation_fn": torch.nn.ReLU,
        }
    },
}

#reseed(seed)
expert_callback = PPOCallback(
    save_freq=5000,
    num_eval_episodes=20,
    save_path=save_path, 
    eval_env=kv_vec_env_1,
    log=log
)
    
with open(log,'a') as file:
    file.writelines([f'{datetime.datetime.now()}>> Starting PPO loop\n'])

#policy_model = PPO('MlpPolicy', kv_vec_env_3, **hyperparameters, device='cpu', verbose=0)
policy_model = PPO(
    'MlpPolicy', 
    kv_vec_env_train, 
    ent_coef = ent_coef,
    #learning_rate=lambda progress_remaining: lr * progress_remaining, 
    **hyperparameters, 
    device='cpu', 
    verbose=0
)
print("Default Learning Rate:", policy_model.lr_schedule(1.0), policy_model.lr_schedule(0.8), policy_model.lr_schedule(0.6), policy_model.lr_schedule(0.4), policy_model.lr_schedule(0.2))
print(policy_model.lr_schedule)
print("Optimizer:", policy_model.policy.optimizer)
print(f"Clip Range: {policy_model.clip_range}")
print(f"Entropy Coefficient (ent_coef): {policy_model.ent_coef}")
print(f"Value Function Coefficient (vf_coef): {policy_model.vf_coef}")
print(f"Discount Factor (gamma): {policy_model.gamma}")

policy_model.learn(total_steps, callback=expert_callback)