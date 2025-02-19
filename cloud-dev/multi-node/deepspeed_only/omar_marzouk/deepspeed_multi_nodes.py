

import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize DeepSpeed and Torch Distributed
rank = int(os.environ.get("RANK", 0))
#world_size = int(os.getenv('WORLD_SIZE', '1'))

if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

# Load model and tokenizer (assumes CPU inference)
MODEL_PATH = "/scratch/user03/1bmodel"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# DeepSpeed inference configuration
ds_config_path = "ds_config.json"
generator.model = deepspeed.init_inference(
    generator.model,
    config=ds_config_path,
    dtype=torch.float32,
    replace_with_kernel_inject=False
)

# Ensure tokenizer has a pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"[Rank {rank}] DeepSpeed Engine Initialized.")

# Verify DeepSpeed Model Parallelism
input_text = "The quick brown fox jumps over the lazy dog."
#inputs = tokenizer(input_text, return_tensors="pt").to("cpu")


print("Before Generation")
#start_time = time.time()
#with torch.no_grad():
#    outputs = ds_engine.generate(
#        **inputs,
#        max_length=30,
#        do_sample=True,
#        temperature=0.8,
#        top_k=50,
#        top_p=0.95
#    )

string = generator(input_text, do_sample=True, min_length=50, max_new_tokens=50)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(string)

print("After Generation")

inference_time = time.time() - start_time
#response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("After Decoding")
print(f"[Rank {rank}] Response: {response_text}")
print(f"[Rank {rank}] Inference Time: {inference_time:.2f} seconds")

# Keep processes alive
if __name__ == "__main__":
    while True:
        time.sleep(60)

