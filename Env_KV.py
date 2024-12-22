import torch
import numpy as np
from utils import release_kv_cache_at_i

class Env_KV:
    def __init__(self,model,tokenizer,max_length,seed=-1):
        self.seed(seed)
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = model.config.n_ctx
        self.max_length = max_length
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
        self.past_kv = release_kv_cache_at_i(past_key_values=self.past_kv,i=action)
        outputs = self.model(self.input_ids, past_key_values=self.past_kv, use_cache=True)
        self.past_kv = outputs.past_key_values
        logits = outputs.logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        next_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        self.input_ids = next_token_id
        self.generated_text += next_token
        self.step += 1
        return self.past_kv, self.reward(action), self.step == self.max_length, None

    def reward(self,action):
        return None

    def reset(self):
        self.generated_text = ""
        self.past_kv = None
        self.input_ids = None
        self.step = 0
    
    def reset_to_input(self,input_text):
        self.generated_text = input_text
        self.input_ids = self.tokenizer(input_text,return_tensors="pt").input_ids.to(self.model.device)
        self.past_kv = self.model(self.input_ids, past_key_values=None, use_cache=True)
        self.past_kv = release_kv_cache_at_i(past_key_values=self.past_kv,i=-1) # i=-1?
        self.step = len(self.past_kv)