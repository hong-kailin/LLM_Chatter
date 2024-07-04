import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-1_8b", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
inputs = tokenizer(["来到美丽的大自然"], return_tensors="pt")
for k, v in inputs.items():
    inputs[k] = v.cuda()
gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
output = model.generate(**inputs, **gen_kwargs)
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print(output)

