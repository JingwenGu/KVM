import torch
import torch.nn.functional as F
import numpy as np
import random
from utils import release_kv, append_kv, truncate_kv

class Env_KV:
    def __init__(self,model,tokenizer,max_length,dataset,seed=-1):
        # expects dataset to be a list of strings
        self.seed(seed)
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = model.config.n_ctx
        self.max_length = max_length
        self.dataset = dataset
        self.reset()
        # self.action_space = np.linspace(self.context_length)
        # self.observation_space = None

    def seed(self,seed):
        self.seed = seed
        if seed != -1:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def step(self,action):
        # return state, reward, done, info
        self.past_kv = release_kv(past_key_values=self.past_kv,i=action)
        outputs = self.model(self.input_ids, past_key_values=self.past_kv, use_cache=True)                    
        logits = outputs.logits
        reward = -F.cross_entropy(logits,self.truth_ids[self.step])
        self.past_kv = append_kv(self.past_kv,self.truth_kv,self.step)
        self.step += 1
        return self.past_kv, reward, self.step == self.max_length, None

    def reset(self):
        self.truth_text = random.choice(self.dataset)
        self.truth_ids = self.tokenizer(self.truth_text,return_tensors="pt").input_ids.to(self.model.device)
        self.truth_kv = self.model(self.truth_ids, past_key_values=None, use_cache=True)
        self.past_kv = None
        self.step = 0
    
    def reset_to_step(self,i):
        self.truth_text = random.choice(self.dataset)
        self.truth_ids = self.tokenizer(self.truth_text,return_tensors="pt").input_ids.to(self.model.device)
        self.truth_kv = self.model(self.truth_ids, past_key_values=None, use_cache=True)
        self.past_kv = truncate_kv(self.truth_kv,i)
        self.step = i