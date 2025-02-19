import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get the process rank from environment variables
rank = int(os.environ.get("RANK", 0))
if not dist.is_initialized():
    dist.init_process_group(backend="c10d")

# Load model and tokenizer (assumes CPU inference)
MODEL_PATH = "/scratch/user03/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

# DeepSpeed inference configuration
ds_config_path = "/scratch/user03/scripts/llm-inference/multi-node/deepspeed_only/ds_config.json"
ds_engine = deepspeed.init_inference(
    model,
    config=ds_config_path,
    dtype=torch.float32,  # CPU inference uses float32
    replace_with_kernel_inject=False,
    tensor_parallel={"tp_size": 2}
)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Only rank 0 starts Ray Serve and deploys the HTTP API.
if rank == 0:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import ray
    from ray import serve

    ray.init(address="auto", ignore_reinit_error=True)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})

    app = FastAPI()

    class QueryRequest(BaseModel):
        input_text: str

    @serve.deployment(num_replicas=1)
    @serve.ingress(app)
    class DeepSpeedLLMService:
        def __init__(self):
            print("Initializing DeepSpeed LLM Service...")
            self.tokenizer = tokenizer
            self.ds_engine = ds_engine

        @app.post("/generate")
        async def generate(self, request: QueryRequest):
            input_text = request.input_text
            inputs = self.tokenizer(input_text, return_tensors="pt").to("cpu")
            start_time = time.time()
            with torch.no_grad():
                outputs = self.ds_engine.generate(
                    **inputs,
                    max_length=50,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
            inference_time = time.time() - start_time
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {
                "response": response_text,
                "inference_time": f"{inference_time:.2f} seconds"
            }

    ds_service = DeepSpeedLLMService.bind()
    serve.run(ds_service, route_prefix="/")

print(f"Process with rank {rank} has initialized the DeepSpeed engine.")

# Keep all processes alive to maintain the distributed group.
if __name__ == "__main__":
    while True:
        time.sleep(60)

