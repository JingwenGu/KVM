import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
import datetime
from utils import release_kv, truncate_kv, append_kv, release_id, append_id

class Env_ID(gym.Env):
    def __init__(self,language_model,tokenizer,dataloader,n_heads=12,max_length=100,dim=64,seed=-1,horizon=1024,render_mode=None,log=None):
        # expects dataloader to be a padded and tokenized
        self.seed(seed)
        self.action_space = gym.spaces.Discrete(max_length)
        self.obs_shape = (dataloader.batch_size,max_length)
        self.observation_space = gym.spaces.Box(low=0, high=tokenizer.vocab_size, shape=self.obs_shape, dtype=np.int64)
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_step = max_length
        self.horizon = horizon
        self.dataloader = dataloader
        self.batch_it = dataloader.__iter__()
        self.batch_counter = 0
        #self.reset()
        self.log = log

    def seed(self,seed):
        self.seed = seed
        if seed != -1:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def step(self,action):
        print(self.length,action)
        # return state, reward, done, info
        self.past_kv = release_kv(past_key_values=self.past_kv,i=action)
        self.curr_ids = release_id(self.curr_ids,action)
        outputs = self.language_model(self.input_ids, past_key_values=self.past_kv, use_cache=True)
        logits = outputs.logits
        reward = -F.cross_entropy(logits[:, -1, :],self.truth_ids[:,self.length]).item()
        print(reward)
        self.input_ids = self.truth_ids[:,self.length:self.length+1]
        self.past_kv = append_kv(self.past_kv,self.truth_kv,self.length)
        #self.past_kv = outputs.past_key_values
        self.curr_ids = append_id(self.curr_ids,self.truth_ids,self.length)
        self.length += 1
        terminate = self.length == min(self.horizon,self.truth_ids.shape[1]) or torch.sum(self.batch['attention_mask'][:,self.length]).item() == 0
        return self.ids_to_obs(self.curr_ids), reward, terminate, False, {}

    def reset(self, *, seed=None, options=None):
        return self.reset_to_step(self.start_step),{}

    def reset_to_step(self,i):
        self.batch_counter += 1
        try:
            self.batch = next(self.batch_it)
        except:
            self.batch_it = self.dataloader.__iter__()
            self.batch = next(self.batch_it)
        self.avg_length = torch.sum(self.batch['attention_mask']).item()/self.obs_shape[0]
        self.truth_ids = self.batch['input_ids']
        outputs = self.language_model(self.truth_ids, past_key_values=None, use_cache=True)
        self.truth_kv = outputs.past_key_values
        self.input_ids = self.truth_ids[:,i:i+1]
        self.curr_ids = self.input_ids[:,:i]
        self.past_kv = truncate_kv(self.truth_kv,i)
        self.length = i
        print(f'reset {self.batch_counter}=========================================================================')
        print(f'avg_length={self.avg_length}')
        with open(self.log,'a') as file:
            file.writelines([f'{datetime.datetime.now()}>> reset {self.batch_counter}, avg_length={self.avg_length}\n'])
        return self.ids_to_obs(self.curr_ids)

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