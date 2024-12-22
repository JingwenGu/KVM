import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset,load_dataset
from stable_baselines3 import PPO
import random
import math
from utils import release_id,release_kv,truncate_kv,append_id,append_kv,pad_obs_tensor,ids_to_obs

def inference_text(policy,language_model,tokenizer,text,n_heads=12,max_length=100,dim=64,seed=-1,horizon=300,log=None):
    device = language_model.device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids[:min(input_ids.shape[-1],max_length-1)]
    outputs = language_model(generated_ids, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = None
    obs_shape = policy.observation_space.shape
    if input_ids.shape[-1] < max_length-1:
        for _ in range(max_length-1-input_ids.shape[-1]):
          logits = outputs.logits
          next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
          outputs = language_model(next_token_id, past_key_values=past_key_values, use_cache=True)
          past_key_values = outputs.past_key_values
          generated_ids = torch.cat([generated_ids, next_token_id], dim=-1) #?
    eos_id = tokenizer(tokenizer.eos_token)['input_ids'][0]
    print(generated_ids.shape)
    complete_ids = generated_ids[:,:]
    while next_token_id != eos_id and complete_ids.shape[-1] <= horizon:
        print(next_token_id)
        logits = outputs.logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        outputs = language_model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1) #?
        complete_ids = torch.cat([complete_ids, next_token_id], dim=-1) #?
        observation = ids_to_obs(generated_ids,obs_shape)
        action = policy.predict(observation)
        print(action)
        print(complete_ids.shape)
        action = action[0]
        past_key_values = release_kv(past_key_values,action)
        generated_ids = release_id(generated_ids,action)
    generated_text = tokenizer.decode(complete_ids[0], skip_special_tokens=True)
    return (generated_text,generated_ids,past_key_values)

def inference_ppl(policy,language_model,tokenizer,batch,n_heads=12,max_length=100,dim=64,seed=-1,horizon=300,log=None):
    avg_length = torch.sum(batch['attention_mask']).item()/1
    i = max_length
    truth_ids = batch['input_ids']
    outputs = language_model(truth_ids, past_key_values=None, use_cache=True)
    truth_kv = outputs.past_key_values
    input_ids = truth_ids[:,i:i+1]
    curr_ids = input_ids[:,:i]
    past_kv = truncate_kv(truth_kv,i)
    length = i
    #eos_id = tokenizer(tokenizer.eos_token)['input_ids'][0]
    obs_shape = policy.observation_space.shape if policy != 'random' and policy != 'slidingwindow' else None
    CE = []
    while length < min(horizon,truth_ids.shape[1]) and torch.sum(batch['attention_mask'][:,length]).item() != 0:
        if policy == 'random':
            action = random.randint(0,length-1)
        elif policy == 'slidingwindow':
            action = 0
        else:
            observation = ids_to_obs(curr_ids,obs_shape)
            action = policy.predict(observation)[0]
        print(length,action)
        # return state, reward, done, info
        past_kv = release_kv(past_key_values=past_kv,i=action)
        curr_ids = release_id(curr_ids,action)
        outputs = language_model(input_ids, past_key_values=past_kv, use_cache=True)
        logits = outputs.logits
        reward = -F.cross_entropy(logits[:, -1, :],truth_ids[:,length]).item()
        print(reward)
        input_ids = truth_ids[:,length:length+1]
        past_kv = append_kv(past_kv,truth_kv,length)
        #self.past_kv = outputs.past_key_values
        curr_ids = append_id(curr_ids,truth_ids,length)
        length += 1
        CE.append(reward)
    avg_CE = sum(CE)/len(CE)
    return CE,sum(CE),avg_CE,math.exp(-avg_CE)

# policy = PPO.load('trained_checkpoints/ID_1211_2.zip')
#policy = PPO.load('trained_checkpoints/ID_1210_1.zip')

# policy_dir = 'trained_ppl_checkpoints/ID_kl_1213_1.zip'
policy_dir = 'trained_checkpoints/ID_1211_2.zip'
# policy_dir = 'slidingwindow'


policy_dir = 'trained_checkpoints/rouge_ID_llama_1213_1.zip'
# policy_dir = 'trained_ppl_checkpoints/ID_ppl_llama_1212_1.zip'
# policy_dir = 'trained_ppl_checkpoints/ID_kl_llama_1213_1.zip'
# policy_dir = 'trained_checkpoints/ID_llama_1212_1.zip'
# policy_dir = 'slidingwindow'
max_length = 100
horizon = 1024
n_eval = 10
model_choice = 'llama'

policy = PPO.load(policy_dir) if policy_dir.find('zip') != -1 else policy_dir

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

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = []
for sentence in dataset['text']:
    ids = tokenizer(sentence, return_tensors="pt").input_ids
    if ids.shape[1] > max_length:
        texts.append(sentence)
    if len(texts) > 1000:
        break
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

it = dataloader.__iter__()
results = []
for i in range(n_eval):
    #generated_text,generated_ids,past_key_values = inference_text(policy,language_model,tokenizer,random.choice(texts))
    CE,sum_CE,avg_CE,PPL = inference_ppl(policy,language_model,tokenizer,next(it))
    print(sum_CE,avg_CE,PPL)
    results.append((CE,sum_CE,avg_CE,PPL))
sum_CE = sum([r[1] for r in results])/len(results)
avg_CE = sum([r[2] for r in results])/len(results)
avg_PPL = sum([r[3] for r in results])/len(results)
print(sum_CE,avg_CE,avg_PPL)