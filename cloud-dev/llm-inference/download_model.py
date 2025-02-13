from transformers import AutoModel, AutoTokenizer

model_name = "openai-community/gpt2-large"  # Change to your model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

save_directory = "/scratch/user03/model"  # Specify where to save
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
