import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from utils import release_kv, truncate_kv, append_kv, kv_to_ndarray, ndarray_to_kv

class Env_KV1(gym.Env):
    def __init__(self,language_model,tokenizer,dataset,batch_size=1,n_heads=12,max_length=100,dim=64,seed=-1,horizon=1024,render_mode=None):
        # expects dataset to be a list of strings
        self.seed(seed)
        self.action_space = gym.spaces.Discrete(max_length)
        self.obs_shape = (language_model.config.n_layer,2,batch_size,n_heads,max_length,dim)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.start_step = max_length
        self.horizon = horizon
        # texts = []
        # for sentence in dataset['text']:
        #   ids = tokenizer(sentence, return_tensors="pt").input_ids
        #   if len(ids[1]) > max_length:
        #     texts.append(sentence)
        # self.dataset = texts
        # Should actually do the above, but to avoid rerunning cells do the below:
        self.dataset = dataset
        self.reset()

    def seed(self,seed):
        self.seed = seed
        if seed != -1:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def step(self,action):
        # return state, reward, done, info
        self.past_kv = release_kv(past_key_values=self.past_kv,i=action)
        outputs = self.language_model(self.input_ids, past_key_values=self.past_kv, use_cache=True)
        logits = outputs.logits
        reward = -F.cross_entropy(logits[:, -1, :],self.truth_ids[:,self.length-1])
        self.input_ids = self.truth_ids[:,:self.length+1]
        self.past_kv = append_kv(self.past_kv,self.truth_kv,self.length)
        #self.past_kv = outputs.past_key_values
        self.length += 1
        return self.kv_to_obs(self.past_kv), reward, self.length == min(self.horizon,self.truth_ids.shape[1]), None, {}

    def reset(self, *, seed=None, options=None):
        # seed = kwargs.get('seed',self.seed)
        # seed = kwargs.get('seed')
        # self.seed(seed)
        print('reset')
        return self.reset_to_step(self.start_step),{}

    def reset_to_step(self,i):
        self.truth_text = random.choice(self.dataset)
        self.truth_ids = self.tokenizer(self.truth_text,return_tensors="pt").input_ids.to(self.language_model.device)
        outputs = self.language_model(self.truth_ids, past_key_values=None, use_cache=True)
        self.truth_kv = outputs.past_key_values
        self.input_ids = self.truth_ids[:,:i]
        self.past_kv = truncate_kv(self.truth_kv,i)
        self.length = i
        return self.kv_to_obs(self.past_kv)

    def pad_obs_tensor(self,kv):
        original_shape = kv.shape
        padded_tensor = torch.zeros(self.obs_shape[2:], dtype=kv.dtype, device=kv.device)
        padded_tensor[..., :original_shape[-2], :] = kv
        return padded_tensor

    def kv_to_obs(self,kv):
        return kv_to_ndarray([(self.pad_obs_tensor(K),self.pad_obs_tensor(V)) for K,V in kv])

    def render(self):
        pass

    def close(self):
        pass

gym.register(id='EnvKV1', entry_point=Env_KV1)