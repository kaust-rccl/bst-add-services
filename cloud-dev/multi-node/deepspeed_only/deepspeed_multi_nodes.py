import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize DeepSpeed and Torch Distributed
rank = int(os.environ.get("RANK", 0))

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["DEEPSPEED_DEBUG"] = "INFO"

if not dist.is_initialized():
    dist.init_process_group(backend="gloo")
print(f"[Rank {rank}] Torch distributed world size: {dist.get_world_size()}")


# Load model and tokenizer (assumes CPU inference)
MODEL_PATH = "/scratch/user03/1bmodel"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

print(f"[Rank {rank}] Model and tokenizer loaded successfully.")
#for name, param in ds_engine.module.named_parameters():
 #   print(f"[Rank {rank}] {name}: {param.shape}, device: {param.device}")

#print(f"[Rank {rank}] DeepSpeed TP Initialized with {ds_engine.config['tp_size']} processes.")


# DeepSpeed inference configuration
ds_config_path = "/scratch/user03/scripts/llm-inference/multi-node/ds_config.json"
ds_engine = deepspeed.init_inference(
    model,
    config=ds_config_path,
    dtype=torch.float32,
    replace_with_kernel_inject=False
)
if ds_engine is None:
    raise RuntimeError(f"[Rank {rank}] DeepSpeed initialization failed!")

print(f"[Rank {rank}] DeepSpeed Engine created successfully: {ds_engine}")

for name, param in ds_engine.module.named_parameters():
    print(f"[Rank {rank}] Parameter: {name}, Shape: {param.shape}, Device: {param.device}")

# Ensure tokenizer has a pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"[Rank {rank}] DeepSpeed Engine Initialized.")

# Verify DeepSpeed Model Parallelism
input_text = "Hello"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

print(f"[Rank {rank}] Initial Input Shape: {inputs['input_ids'].shape}")
print(f"[Rank {rank}] MASTER_ADDR: {os.getenv('MASTER_ADDR')}, MASTER_PORT: {os.getenv('MASTER_PORT')}")

#if dist.get_world_size() > 1:
 #   inputs = {k: v.chunk(dist.get_world_size(), dim=-1)[dist.get_rank()] for k, v in inputs.items()}
  #  print(f"[Rank {rank}] Adjusted Input Shape: {inputs['input_ids'].shape}")

inputs = {k: v.to(dtype=torch.float32) if k != "input_ids" else v.to(dtype=torch.int64) for k, v in inputs.items()}

print(f"[Rank {rank}] Converted Input to Float32: {inputs['input_ids'].dtype}")

dist.barrier()
print(f"[Rank {rank}] All processes synchronized before generate()")

start_time = time.time()
with torch.no_grad():
    outputs = ds_engine.module.generate(
        **inputs,
        max_length=10,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

#with torch.no_grad():
 #   outputs = ds_engine.module(**inputs)
  #  print("in shaa allah no faults.")


inference_time = time.time() - start_time
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"[Rank {rank}] Response: {response_text}")
print(f"[Rank {rank}] Inference Time: {inference_time:.2f} seconds")

# Keep processes alive
if __name__ == "__main__":
    while True:
        time.sleep(60)

