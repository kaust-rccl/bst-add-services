from transformers import AutoModelForCausalLM

model_path = "/scratch/user03/1bmodel"
model = AutoModelForCausalLM.from_pretrained(model_path)

