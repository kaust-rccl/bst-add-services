import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Torch distributed (using Gloo backend for CPU)
if not dist.is_initialized():
    dist.init_process_group(backend="gloo")
rank = dist.get_rank()
print(f"[Rank {rank}] Torch distributed world size: {dist.get_world_size()}")

# Load model and tokenizer (CPU inference)
MODEL_PATH = "/scratch/user03/1bmodel"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Ensure a pad token exists (important for some tokenizers)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")
print(f"[Rank {rank}] Model and tokenizer loaded successfully.")

# DeepSpeed Inference configuration for CPU (note: using init_inference)
ds_config = {
    "tensor_parallel": {"tp_size": 4}
}
# Use init_inference for inference-only mode (avoids train_batch_size error)
ds_engine = deepspeed.init_inference(
    model=model,
    config=ds_config,
    dtype=torch.float32, 
    replace_with_kernel_inject=False  # Set True if you wish to use kernel injection
)

#ds_engine, optimizer, _, _ = deepspeed.initialize(
#model=model,
#    config=ds_config,
#    model_parameters=model.parameters(),
#)

print(f"[Rank {rank}] DeepSpeed Inference Engine created successfully.")
total_params = sum(p.numel() for p in ds_engine.module.parameters())
print(f"[Rank {rank}] Number of parameters in shard: {total_params}")

# Example Inference: Generate text
input_text = "#Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

dist.barrier()

with torch.no_grad():
    outputs = ds_engine.module.generate(**inputs, max_length=32)
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"[Rank {rank}] Response: {response_text}")

# Keep process alive if needed
if __name__ == "__main__":
    while True:
        time.sleep(60)

