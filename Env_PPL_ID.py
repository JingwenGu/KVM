import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import evaluate
import random
import datetime
from utils import release_kv, truncate_kv, append_kv, release_id, append_id, ids_to_obs

class Env_PPL_ID(gym.Env):
    def __init__(self,language_model,tokenizer,dataloader,n_heads=12,max_length=100,dim=64,seed=-1,horizon=1024,repeat_len=20,render_mode=None,log=None):
        # expects dataloader to be a padded and tokenized
        self.seed(seed)
        self.action_space = gym.spaces.Discrete(max_length)
        self.obs_shape = (dataloader.batch_size,max_length)
        self.observation_space = gym.spaces.Box(low=0, high=tokenizer.vocab_size, shape=self.obs_shape, dtype=np.int64)
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.eos_id = self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        self.device = language_model.device
        self.max_length = max_length
        self.start_step = max_length
        self.horizon = horizon
        self.repeat_len = repeat_len
        self.dataloader = dataloader
        self.batch_it = dataloader.__iter__()
        self.batch_counter = 0
        self.rouge = evaluate.load("rouge")
        self.log = log

    def seed(self,seed):
        self.seed = seed
        if seed != -1:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def step(self,action):
        print(self.length,action)
        # print(self.input_ids.shape)
        # print(self.next_token_id)
        self.past_kv = release_kv(self.past_kv,action)
        self.curr_ids = release_id(self.curr_ids,action)
        self.outputs = self.language_model(self.next_token_id, past_key_values=self.past_kv, use_cache=True)
        true_outputs = self.language_model(self.input_ids)
        logits = self.outputs.logits
        self.next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        true_next_token_id = torch.argmax(true_outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        # print(self.next_token_id,true_next_token_id)
        reward = -F.cross_entropy(logits[:, -1, :],true_next_token_id[0]).item()
        print(reward)
        self.past_kv = self.outputs.past_key_values
        self.curr_ids = torch.cat([self.curr_ids, self.next_token_id], dim=-1) #?
        self.input_ids = torch.cat([self.input_ids, self.next_token_id], dim=-1) #?
        #observation = ids_to_obs(self.curr_ids,self.obs_shape)

        self.length += 1
        terminate = self.length == self.horizon or self.next_token_id == self.eos_id or self.repeat()
        return self.ids_to_obs(self.curr_ids), reward, terminate, False, {}
    
    def repeat(self):
        for i in range(self.input_ids.shape[-1]-self.repeat_len):
            if (self.input_ids[:,i:i+self.repeat_len] == self.input_ids[:,-self.repeat_len:]).all().item():
                print('repeat')
                return True
        return False

    def reset(self, *, seed=None, options=None):
        # print(self.max_length)
        self.batch_counter += 1
        try:
            self.batch = next(self.batch_it)
        except:
            self.batch_it = self.dataloader.__iter__()
            self.batch = next(self.batch_it)
        self.input_ids = self.batch['input_ids']
        # self.input_ids = self.tokenizer(self.batch['dialogue'], return_tensors="pt").input_ids.to(self.device)
        self.input_ids = self.input_ids[:,:min(self.input_ids.shape[-1],self.max_length-1)]
        self.outputs = self.language_model(self.input_ids, past_key_values=None, use_cache=True)
        self.past_kv = self.outputs.past_key_values
        self.next_token_id = None
        # print(self.input_ids.shape)
        for _ in range(self.max_length-1-self.input_ids.shape[-1]):
            logits = self.outputs.logits
            self.next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            self.outputs = self.language_model(self.next_token_id, past_key_values=self.past_kv, use_cache=True)
            self.past_kv = self.outputs.past_key_values
            self.input_ids = torch.cat([self.input_ids, self.next_token_id], dim=-1) #?
        if self.next_token_id == None:
            logits = self.outputs.logits
            self.next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        # print(self.input_ids.shape)
        self.curr_ids = self.input_ids[:,:]
        self.length = self.max_length
        print(f'reset {self.batch_counter}=========================================================================')
        with open(self.log,'a') as file:
            file.writelines([f'{datetime.datetime.now()}>> reset {self.batch_counter}\n'])
        return self.ids_to_obs(self.curr_ids), {}

    def pad_obs_tensor(self,ids):
        original_shape = ids.shape
        padded_tensor = torch.zeros(self.obs_shape, dtype=ids.dtype, device=ids.device)
        padded_tensor[..., :original_shape[-1]] = ids
        return padded_tensor

    def ids_to_obs(self,ids):
        return self.pad_obs_tensor(ids).detach().numpy()

    def render(self):
        pass

    def close(self):
        pass