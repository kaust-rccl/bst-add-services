import os
import time
import torch
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get process rank; all processes do this.
rank = int(os.environ.get("RANK", 0))
if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

# All processes initialize the model shard.
MODEL_PATH = "/scratch/user03/model"
# Initialize tokenizer and model (for CPU inference)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

ds_config_path = "/scratch/user03/llm-inference/multi-node/ds_config.json"
ds_engine = deepspeed.init_inference(
    model,
    config=ds_config_path,
    dtype=torch.float32,  # CPU inference uses float32
    replace_with_kernel_inject=False
)

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

    # Create a FastAPI app to use with Ray Serve.
    app = FastAPI()

    # Define the request schema.
    class QueryRequest(BaseModel):
        input_text: str

    # Define the Ray Serve deployment wrapper.
    @serve.deployment(num_replicas=1)
    @serve.ingress(app)
    class DeepSpeedLLMService:
        def __init__(self):
            print("Loading tokenizer and model with DeepSpeed (wrapper)...")
            # Use the already initialized tokenizer and DeepSpeed engine.
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

    # Bind and run the Ray Serve deployment on rank 0.
    ds_service = DeepSpeedLLMService.bind()
    serve.run(ds_service, route_prefix="/")

print(f"Process with rank {rank} initialized DeepSpeed engine.")

if __name__ == "__main__":
    # All processes (including non-rank 0) keep running to maintain the distributed group.
    while True:
        time.sleep(60)

