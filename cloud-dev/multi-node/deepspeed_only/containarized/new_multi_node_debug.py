import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import psutil

def debug_print(*args, **kwargs):
    print("[DEBUG]", *args, **kwargs)

# Debug: Process and host info
debug_print("PID:", os.getpid())
debug_print("Hostname:", socket.gethostname())
debug_print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
debug_print("MASTER_ADDR:", os.environ.get("MASTER_ADDR", None))
debug_print("MASTER_PORT:", os.environ.get("MASTER_PORT", None))

# Determine rank from environment variables set by torch.distributed.run
rank = int(os.environ.get("RANK", 0))
debug_print("Process rank from env:", rank)

# Initialize process group if not already done
if not dist.is_initialized():
    debug_print("Initializing process group with gloo backend...")
    dist.init_process_group(backend="gloo")
    debug_print("Process group initialized.")
else:
    debug_print("Process group already initialized.")

# Debug: Print distributed world size
world_size = dist.get_world_size()
debug_print("World size:", world_size)

# Load model and tokenizer (assumes CPU inference)
MODEL_PATH = "/scratch/user03/1bmodel"  # Update as needed
debug_print("Loading model from:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
debug_print("Tokenizer loaded. Vocabulary size:", len(tokenizer))
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")
debug_print("Model loaded and moved to CPU.")

# DeepSpeed inference configuration (ensure the path is correct)
ds_config_path = "/scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/new_ds_config.json"
debug_print("Loading DeepSpeed config from:", ds_config_path)
try:
    with open(ds_config_path, "r") as f:
        ds_config_contents = f.read()
    debug_print("DeepSpeed config contents:", ds_config_contents)
except Exception as e:
    debug_print("Error reading DeepSpeed config:", e)

# Initialize DeepSpeed inference engine with additional debugging
try:
    debug_print("Initializing DeepSpeed inference engine...")
    ds_engine = deepspeed.init_inference(
        model,
        config=ds_config_path,
        dtype=torch.float32,
        replace_with_kernel_inject=False
    )
    debug_print("DeepSpeed inference engine initialized successfully.")
except Exception as e:
    debug_print("Error during DeepSpeed inference initialization:", e)
    raise

# Ensure pad token is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    debug_print("Pad token not set; using eos_token_id:", tokenizer.eos_token_id)

print(f"[Rank {rank}] DeepSpeed Engine Initialized.")

# Run inference (each process can run the same prompt or different prompts)
test_prompt = "What is AI?"
debug_print("Running inference on prompt:", test_prompt)
inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")
debug_print("Prepared inputs:", inputs)

start_time = time.time()
try:
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
    debug_print("Generation completed.")
except Exception as e:
    debug_print("Error during generation:", e)
    raise

print(f"[Rank {rank}] Inference time: {end_time - start_time:.2f} seconds")
generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"[Rank {rank}] Q: {test_prompt}\nA: {generated_answer}")

# Post-inference: Print system resource usage
cpu_percent = psutil.cpu_percent(interval=1)
mem_usage = psutil.virtual_memory().used / 1024**3
debug_print("Post-inference CPU usage:", cpu_percent, "%")
debug_print("Post-inference Memory usage:", mem_usage, "GB")

# Optionally keep the process alive for debugging:
if __name__ == "__main__":
    while True:
        time.sleep(60)

