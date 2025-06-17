import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
# from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers import Qwen2Config

model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
# tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]