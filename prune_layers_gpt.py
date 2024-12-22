import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class ScriptArguments:
    model_name:str = field(default="gpt2-medium", metadata={"help": "the teacher model name"})
    start_prune:int = field(default=0, metadata={"help": "first pruned layer index, inclusive"})
    end_prune:int = field(default=0, metadata={"help": "last pruned layer index, inclusive"})
    output_dir:str = field(default="", metadata={"help": "output directory"})
parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_name_or_path = script_args.model_name
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# num_layers = len(model.transformer.h)
# print(f"Number of original layers: {num_layers}")

for _ in range(script_args.end_prune-script_args.start_prune+1):
    del model.transformer.h[script_args.start_prune]

os.makedirs(script_args.output_dir,exist_ok=True)
model.config.num_hidden_layers -= (script_args.end_prune-script_args.start_prune+1)
model.save_pretrained(script_args.output_dir)