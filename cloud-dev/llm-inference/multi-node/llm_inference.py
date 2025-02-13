import time
import torch
import deepspeed
import ray
from ray import serve
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Start Ray
#ray.init(ignore_reinit_error=True)
ray.init(address="auto", ignore_reinit_error=True)
# Start Ray Serve
serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})

# Define request structure
class QueryRequest(BaseModel):
    input_text: str

@serve.deployment(num_replicas=1)
@serve.ingress(app)  # Attach FastAPI to Ray Serve
class DeepSpeedLLMService:
    def __init__(self):
        print("Loading tokenizer and model with DeepSpeed...")

        # Load tokenizer
        MODEL_PATH = "/scratch/user03/model"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cpu")

        # Load DeepSpeed config
        ds_config_path = "/scratch/user03/llm-inference/multi-node/ds_config.json"

        # Initialize DeepSpeed inference engine
        self.ds_engine = deepspeed.init_inference(
            model,
            config=ds_config_path,
            dtype=torch.float32,  # Use float32 for CPU
            replace_with_kernel_inject=False
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @app.post("/generate")
    async def generate(self, request: QueryRequest):
        """Handles text generation requests using DeepSpeed-optimized model."""
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
        end_time = time.time()

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "response": response_text,
            "inference_time": f"{end_time - start_time:.2f} seconds"
        }

# Bind and deploy the service
ds_service = DeepSpeedLLMService.bind()
serve.run(ds_service, route_prefix="/")

if __name__ == "__main__":
    print("DeepSpeed LLM service is running at http://localhost:8000")
    while True:
        time.sleep(60)

