import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Determine rank from environment variables set by torch.distributed.run
rank = int(os.environ.get("RANK", 0))

# Initialize process group if not already done
if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

# Load model and tokenizer (assumes CPU inference)
MODEL_PATH = "/scratch/user03/gptj"  # Update as needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

# DeepSpeed inference configuration (ensure the path is correct)
ds_config_path = "/scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/ds_config.json"
ds_engine = deepspeed.init_inference(
    model,
    config=ds_config_path,
    dtype=torch.float32,
    replace_with_kernel_inject=False
)

# Ensure pad token is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"[Rank {rank}] DeepSpeed Engine Initialized.")

# Run inference (each process can run the same prompt or different prompts)
test_prompt = "What is AI?"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")

start_time = time.time()
with torch.no_grad():
    outputs = ds_engine.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
end_time = time.time()

print(f"[Rank {rank}] Inference time: {end_time - start_time:.2f} seconds")
generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"[Rank {rank}] Q: {test_prompt}\nA: {generated_answer}")

# Optionally keep the process alive for debugging:
if __name__ == "__main__":
    while True:
        time.sleep(60)

