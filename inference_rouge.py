import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset,load_dataset
from stable_baselines3 import PPO
import random
import math
import evaluate
from utils import release_id,release_kv,truncate_kv,append_id,append_kv,pad_obs_tensor,ids_to_obs

def repeat(input_ids,repeat_len):
    for i in range(input_ids.shape[-1]-repeat_len):
        if (input_ids[:,i:i+repeat_len] == input_ids[:,-repeat_len:]).all().item():
            print('repeat')
            return True
    return False

def inference_text(policy,language_model,tokenizer,text,n_heads=12,max_length=100,dim=64,seed=-1,horizon=300,log=None):
    device = language_model.device
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids[:min(input_ids.shape[-1],max_length-1)]
    outputs = language_model(generated_ids, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = None
    obs_shape = policy.observation_space.shape if policy != 'slidingwindow' and policy != 'random' else None
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
    while next_token_id != eos_id and complete_ids.shape[-1] <= horizon and not repeat(generated_ids,10):
        # print(next_token_id)
        logits = outputs.logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        outputs = language_model(next_token_id, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1) #?
        complete_ids = torch.cat([complete_ids, next_token_id], dim=-1) #?
        if policy == 'slidingwindow':
            action = 0
        elif policy == 'random':
            action = random.randint(0,complete_ids.shape[-1]-1)
        else:
            observation = ids_to_obs(generated_ids,obs_shape)
            action = policy.predict(observation)
            # print(action)
            # print(complete_ids.shape)
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

# policy_dir = 'trained_checkpoints/rouge_ID_1213_1.zip'
# policy_dir = 'trained_ppl_checkpoints/ID_ppl_1213_1.zip'
# policy_dir = 'trained_ppl_checkpoints/ID_kl_1213_1.zip'
# policy_dir = 'trained_checkpoints/ID_1211_2.zip'

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

dataset_sam = load_dataset("samsum")
new_dialogue = []
new_summary = []
for dialogue,summary in zip(dataset_sam['train']['dialogue'],dataset_sam['train']['summary']):
  ids = tokenizer(dialogue, return_tensors="pt").input_ids
#   print(ids.shape)
  if ids.shape[1] <= max_length:
    new_dialogue.append(dialogue)
    new_summary.append(summary)
new_dataset = Dataset.from_dict({'dialogue': new_dialogue, 'summary': new_summary})

results = []
rouge = evaluate.load("rouge")
for i in range(n_eval):
    #generated_text,generated_ids,past_key_values = inference_text(policy,language_model,tokenizer,random.choice(texts))
    idx = random.randint(0,len(new_dialogue))
    #prompt = f'Summarize the following dialogue:\"{new_dialogue[idx]}\" Summary:'
    prompt = new_dialogue[idx]
    print(prompt)
    generated_text,generated_ids,past_key_values = inference_text(policy,language_model,tokenizer,prompt)
    print(generated_text)
    res = rouge.compute(
        predictions=[generated_text],
        references=[new_summary[idx]]
    )
    results.append(res)
    print(res)
r1 = sum([r['rouge1'] for r in results])/len(results)
r2 = sum([r['rouge2'] for r in results])/len(results)
rl = sum([r['rougeL'] for r in results])/len(results)
rls = sum([r['rougeLsum'] for r in results])/len(results)
print(r1,r2,rl,rls)