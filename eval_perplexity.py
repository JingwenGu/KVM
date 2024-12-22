import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
import tqdm
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_dir: Optional[str] = field(default="gpt2-medium", metadata={"help": "the model directory"})
    tokenizer_name: Optional[str] = field(default="gpt2-medium", metadata={"help": "the tokenizer name"})
    quantized: Optional[int] = field(default=0, metadata={"help": "whether the model is quantized, 1 if quantized"})
    # dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    sample_size: Optional[int] = field(default=100, metadata={"help": "number of samples"})
    # batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    # save_steps: Optional[int] = field(default=1000, metadata={"help": "# steps to save the model"})
    # log_steps: Optional[int] = field(default=10, metadata={"help": "# steps to log the model"})
    # output_dir: Optional[str] = field(default="", metadata={"help": "n steps to save the model"})
    # wandb_name: Optional[str] = field(default="", metadata={"help": "wandb project name"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

print("loaded dataset")

# Load pre-trained GPT-2 model and tokenizer
model_name = script_args.model_dir
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)

print("loaded model")

# Ensure model is in evaluation mode and move it to the appropriate device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if script_args.quantized == 0:
    model.to(device)

# Function to calculate the perplexity
def compute_perplexity(model, tokenizer, dataset):
    total_loss = 0.0
    total_length = 0

    max_iter = script_args.sample_size
    counter = 0

    for i, example in tqdm.tqdm(enumerate(dataset)):
        #print(f"sample {i}")
        # Tokenize input text
        if len(example['text']) < 50:
            continue

        inputs = tokenizer(example['text'], return_tensors='pt', max_length=1024, truncation=True)
        input_ids = inputs.input_ids.to(device)

        # Generate model outputs and calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # Accumulate total loss and token count
        total_loss += loss * input_ids.shape[1]
        total_length += input_ids.shape[1]

        # print(f"prompt: {example['text']}")
        # print(f"output: {tokenizer.decode(model.generate(
        #     input_ids, 
        #     max_length=100, 
        #     num_return_sequences=1,
        #     repetition_penalty=1.2,
        #     temperature=0.7,)[0], skip_special_tokens=True)}")

        counter += 1
        if counter > max_iter:
            break

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_length
    perplexity = np.exp(avg_loss)

    return perplexity

# Calculate perplexity
perplexity = compute_perplexity(model, tokenizer, dataset)
print(f'Perplexity: {perplexity}')

peak_memory_allocated = torch.cuda.max_memory_allocated(device)
peak_memory_reserved = torch.cuda.max_memory_reserved(device)

print(f"Peak memory allocated: {peak_memory_allocated / 1024**2} MB")
print(f"Peak memory reserved: {peak_memory_reserved / 1024**2} MB")