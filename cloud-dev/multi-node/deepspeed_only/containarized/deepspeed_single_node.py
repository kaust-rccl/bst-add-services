import time
import psutil
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA:", torch.cuda.is_available())
MODEL_PATH = "/scratch/user03/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

# DeepSpeed inference configuration
ds_config_path = "/scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/containarized/ds_config.json"


ds_engine = deepspeed.init_inference(
    model,
    config=ds_config_path,
    dtype=torch.float32,
    replace_with_kernel_inject=False
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

test_prompt = "What is AI?"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cpu")

start_time = time.time()
with torch.no_grad():
    outputs = ds_engine.generate(**inputs,
                                 max_length=50,
                                 do_sample=True,
                                 temperature=0.8,
                                 top_k=50,
                                 top_p=0.95
                                 )
end_time = time.time()

print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB")

generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Q: {test_prompt}\nA: {generated_answer}")
