import logging
logging.basicConfig(level=logging.INFO)

from transformers import AutoTokenizer, Qwen2ForCausalLM
# from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers import Qwen2Config

print("Script starting...")

print("Loading model Qwen2ForCausalLM.from_pretrained(\"Qwen/Qwen2.5-1.5B\")...")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
print("Model loaded.")

print("Loading tokenizer AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-1.5B\")...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
print("Tokenizer loaded.")

prompt = "Hello, my name is"
print(f"Processing prompt: '{prompt}'...")
inputs = tokenizer(prompt, return_tensors="pt")
print("Prompt processed into inputs.")
print(f"Input IDs: {inputs.input_ids}")

# If you are using a GPU, ensure inputs are on the same device as the model
# Example:
# if torch.cuda.is_available() and next(model.parameters()).is_cuda:
#     print("Moving inputs to CUDA device.")
#     inputs = {k: v.to('cuda') for k, v in inputs.items()}

print("Calling model.generate(...)...")
generate_ids = model.generate(inputs.input_ids, max_length=30)
print("model.generate(...) finished.")

output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"Generated text: {output_text}")
print("Script finished.")
