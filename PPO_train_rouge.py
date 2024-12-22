env_choice = 'id'
model_choice = 'gpt2'
log = 'trained_checkpoints/log_rouge_1213_id-1.txt'
save_path = 'trained_checkpoints/rouge_ID_1213_1'
total_steps = 1500000
lr = 3e-4
reward_lr = False
ent_coef = 0.01

max_length = 100
horizon = 700
repeat_len = 20

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

tokenizer = AutoTokenizer.from_pretrained('gpt2')
language_model = AutoModelForCausalLM.from_pretrained('gpt2')

if model_choice == 'llama':
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

dataset_sam = load_dataset("samsum")
new_dialogue = []
new_summary = []
for dialogue,summary in zip(dataset_sam['train']['dialogue'],dataset_sam['train']['summary']):
  ids = tokenizer(dialogue, return_tensors="pt").input_ids
  if ids.shape[1] <= max_length:
    new_dialogue.append(dialogue)
    new_summary.append(summary)
new_dataset = Dataset.from_dict({'dialogue': new_dialogue, 'summary': new_summary})

dataloader = torch.utils.data.DataLoader(
    new_dataset, 
    batch_size=1,
    shuffle=True,
)

# gym.register(id='EnvKV', entry_point='Env_KV:Env_KV')
gym.register(id='EnvRougeID', entry_point='Env_rouge_ID:Env_rouge_ID')
env_kwargs = {
    'language_model': language_model, 
    'tokenizer': tokenizer, 
    'dataloader': dataloader, 
    'max_length': max_length, 
    'seed': -1, 
    'horizon': horizon,
    'repeat_len': repeat_len,
    'log': log
}

if env_choice == 'kv':
    kv_vec_env_1 = make_vec_env('EnvKV', n_envs=1, env_kwargs=env_kwargs)
    kv_vec_env_train = make_vec_env('EnvKV', n_envs=1, env_kwargs=env_kwargs)
elif env_choice == 'id':
    kv_vec_env_1 = make_vec_env('EnvRougeID', n_envs=1, env_kwargs=env_kwargs)
    kv_vec_env_train = make_vec_env('EnvRougeID', n_envs=1, env_kwargs=env_kwargs)

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
    log=log,
    reward_lr=reward_lr
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