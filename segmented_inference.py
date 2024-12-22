import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
#from torch.quantization import convert
import time
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
from utils import release_kv

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_dir: Optional[str] = field(default="gpt2-medium", metadata={"help": "the teacher model name"})
    tokenizer_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the tokenizer name"})
    quantized: Optional[int] = field(default=0, metadata={"help": "whether the model is quantized, 1 if quantized"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_dir

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)

model.eval()
#convert(model,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if script_args.quantized == 0:
    model.to(device)


prompt = "Once upon a time, in a galaxy far, far away,"

# with torch.no_grad():
#     print("input<<", prompt)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to(device)
#     outputs = model.generate(
#         input_ids, 
#         max_new_tokens=128, 
#         num_return_sequences=1,
#         repetition_penalty=2.0,
#         temperature=0.5,
#     )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("output>>", generated_text)
#==========================================================================
# print(prompt)
# input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to(device)
# outputs = model(input_ids,use_cache=True)
# kv = outputs.past_key_values
# print(len(kv),len(kv[0]),kv[0][0].shape)
# token_id = torch.argmax(outputs.logits[:,-1,:],dim=-1)
# token = tokenizer.decode(token_id)
# print(token)
# with torch.no_grad():
#     for _ in range(10):
#         prompt += token
#         input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to(device)
#         outputs = model(input_ids,use_cache=True)
#         kv = outputs.past_key_values
#         print(len(kv),len(kv[0]),kv[0][0].shape)
#         token_id = torch.argmax(outputs.logits[:,-1,:],dim=-1)
#         token = tokenizer.decode(token_id)
#         print(token)
#==========================================================================
prompt = "Summarize the following sentence: \"The quick brown fox jumped over the lazy grey dog who was sleeping and snoring.\" Summary: "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Cache initial tokens and prepare for streaming
output_text = prompt
past_key_values = None

# Define the max tokens we want to generate
max_new_tokens = 50

# Token-by-token generation loop
for _ in range(max_new_tokens):
    # Generate the next token logits with kv-cache reuse
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Update cache
    i = -1

    # print(input_ids)
    # past_key_values = release_kv(past_key_values,i)
    # print(len(past_key_values),len(past_key_values[0]),past_key_values[0][0].shape,past_key_values[0][1].shape)

    #print(len(past_key_values1),len(past_key_values1[0]),past_key_values1[0][0].shape,past_key_values1[0][1].shape)

    # Get the most likely next token (top-1)
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
    
    # Add the generated token to the output text
    output_text += next_token
    #print(next_token, end="", flush=True)  # Stream token by token

    # Update input_ids for the next loop iteration
    input_ids = next_token_id

print("\nGenerated text:", output_text)