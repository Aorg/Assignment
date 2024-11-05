import torch,os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
repo_id = "internlm/internlm2_5-1_8b"
local_dir = f"{repo_id.split('/')[1]}"
os.makedirs(local_dir, exist_ok=True)
snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir
    )
tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_dir, torch_dtype=torch.float16, trust_remote_code=True)
model = model.eval()

inputs = tokenizer(["A beautiful flower"], return_tensors="pt")
gen_kwargs = {
    "max_length": 128,
    "top_p": 0.8,
    "temperature": 0.8,
    "do_sample": True,
    "repetition_penalty": 1.0
}

# 以下内容可选，如果解除注释等待一段时间后可以看到模型输出
output = model.generate(**inputs, **gen_kwargs)
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print(output)