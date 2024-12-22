import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
#from torch.quantization import convert
import time
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

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


input_prompts = ["Once upon a time, in a galaxy far, far away,",
                 "A humanoid and an electric sheep walk into a bar,",
                 "I was blind, but now"]

start = time.perf_counter()
with torch.no_grad():
    for prompt in input_prompts:
        print("input<<", prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=128, 
            num_return_sequences=1,
            repetition_penalty=2.0,
            temperature=0.5,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("output>>", generated_text)
end = time.perf_counter()
print(f'Average latency={(end-start)/len(input_prompts)}')