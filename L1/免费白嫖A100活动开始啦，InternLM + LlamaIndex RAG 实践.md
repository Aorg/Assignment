![1729033461435](image/提示词工程课程/1729033461435.png)

# 内容来源：

[Docs](https://aicarrier.feishu.cn/wiki/OWTqwyjEkiGJbCkxib5cGV8Wn1d "Docs")

# 前置知识：

## 检索增强生成（Retrieval Augmented Generation，RAG）

![](https://i-blog.csdnimg.cn/direct/992154459a5e4939a0b5edb93ec74643.png)![]()**编辑**

## LlamaIndex

LlamaIndex 是一个上下文增强的 LLM 框架，旨在通过将其与特定上下文数据集集成，增强大型语言模型（LLMs）的能力。

## `xtuner`

书生集成的微调,测试大模型平台

# 环境搭建：

## 创建环境

服务器已经预设好了conda环境，输入书生服务器封装bash代码运行，激活环境，如下：

```bash
studio-conda -t llamaindex -o pytorch-2.1.2
conda activate llamaindex 
pip install llama-index==0.10.38 llama-index-llms-huggingface==0.2.0 "transformers[torch]==4.41.1" "huggingface_hub[inference]==0.23.1" huggingface_hub==0.23.1 sentence-transformers==2.7.0 sentencepiece==0.2.0

cd ~
mkdir llamaindex_demo
mkdir model
cd ~/llamaindex_demo
touch download_hf.py
vim download_hf.py
```

![]()

![](https://i-blog.csdnimg.cn/direct/246aab2d5ee54d3792a48ad5263fbadb.png)![]()**编辑**

## 下载RAG模型：

键入I，表示输入，在download_hf.py输入：

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
```

![]()

运行下载：

```bash
conda activate llamaindex
python download_hf.py
```

![]()

## 下载nltk资源：

```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

![]()

## LlamaIndex HuggingFaceLLM

```bash
cd ~/model
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/ ./
cd ~/llamaindex_demo
touch llamaindex_internlm.py
vim llamaindex_internlm.py
#点击i 复制一下代码
#from llama_index.llms.huggingface import HuggingFaceLLM
#from llama_index.core.llms import ChatMessage
#llm = HuggingFaceLLM(
#    model_name="/root/model/internlm2-chat-1_8b",
#    tokenizer_name="/root/model/internlm2-chat-1_8b",
#    model_kwargs={"trust_remote_code":True},
#    tokenizer_kwargs={"trust_remote_code":True}
#)

#rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
#print(rsp)

#输入完成
#点击ESC
#：wq
#保存

conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_internlm.py
```

![]()

这里在测试使用大模型是否正常输出,并且跟后续加入RAG后效果对比

```bash
conda activate llamaindex
pip install llama-index-embeddings-huggingface llama-index-embeddings-instructor
cd ~/llamaindex_demo
mkdir data
cd data
git clone https://github.com/InternLM/xtuner.git
mv xtuner/README_zh-CN.md ./
cd ~/llamaindex_demo
touch llamaindex_RAG.py
#写入下面python代码
conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_RAG.py
```

![]()

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

embed_model = HuggingFaceEmbedding(
    model_name="/root/model/sentence-transformer"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
Settings.llm = llm

documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("xtuner是什么?")

print(response)
```

![]()

# 关卡任务

完成以下任务，并将实现过程记录截图：

* 通过 llamaindex 运行 InternLM2 1.8B，询问“你是谁”，将运行结果截图。
* 通过 llamaindex 实现知识库检索，询问两个问题将运行结果截图。
  * 问题1：xtuner是什么?
  * 问题2：xtuner支持那些模型？

完成作业10%RAG是不够的，但是都到这里了 直接向助教申请30%资源：

```bash
/root/.conda/envs/llamaindex/bin/python /root/data/data/llamaindex_RAG.py
/root/.conda/envs/llamaindex/lib/python3.10/site-packages/pydantic/_internal/_fields.py:161: UserWarning: Field "model_id" has conflict with protected namespace "model_".

You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
  warnings.warn(
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:39<00:00, 19.95s/it]
Some parameters are on the meta device device because they were offloaded to the cpu.
Traceback (most recent call last):
  File "/root/data/data/llamaindex_RAG.py", line 23, in <module>
    response = query_engine.query("xtuner是什么?")
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/base/base_query_engine.py", line 52, in query
    query_result = self._query(str_or_query_bundle)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/query_engine/retriever_query_engine.py", line 190, in _query
    response = self._response_synthesizer.synthesize(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/response_synthesizers/base.py", line 241, in synthesize
    response_str = self.get_response(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/response_synthesizers/compact_and_refine.py", line 43, in get_response
    return super().get_response(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/response_synthesizers/refine.py", line 183, in get_response
    response = self._give_response_single(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/response_synthesizers/refine.py", line 238, in _give_response_single
    program(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/response_synthesizers/refine.py", line 84, in __call__
    answer = self._llm.predict(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/llms/llm.py", line 438, in predict
    response = self.complete(formatted_prompt, formatted=True)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/instrumentation/dispatcher.py", line 230, in wrapper
    result = func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/core/llms/callbacks.py", line 429, in wrapped_llm_predict
    f_return_val = f(_self, *args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/llama_index/llms/huggingface/base.py", line 358, in complete
    tokens = self._model.generate(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/transformers/generation/utils.py", line 1758, in generate
    result = self._sample(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/transformers/generation/utils.py", line 2397, in _sample
    outputs = self(
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/accelerate/hooks.py", line 169, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/root/.cache/huggingface/modules/transformers_modules/internlm2-chat-1_8b/modeling_internlm2.py", line 1060, in forward
    logits = self.output(hidden_states)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/accelerate/hooks.py", line 169, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/root/.conda/envs/llamaindex/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 316.00 MiB. GPU 0 has a total capacty of 7.99 GiB of which 198.00 MiB is free. Process 1736952 has 35.84 GiB memory in use. Process 364582 has 7.80 GiB memory in use. Of the allocated memory 7.25 GiB is allocated by PyTorch, and 63.81 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

![]()

有RAG和没RAG结果对比：

![](https://i-blog.csdnimg.cn/direct/5f6239bc8aeb428685d8741e834e3748.png)![]()**编辑**

# 任务截图

将任务写入py

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

embed_model = HuggingFaceEmbedding(
    model_name="/root/model/sentence-transformer"
)

Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
Settings.llm = llm
rsp = llm.chat(messages=[ChatMessage(content="你是谁？")])
print("你是谁？",rsp)
documents = SimpleDirectoryReader("/root/data/data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("xtuner是什么?")
print("xtuner是什么?",response)
response = query_engine.query("xtuner支持哪些模型")
print("xtuner支持哪些模型",response)
```

![]()

你是谁？

![](https://i-blog.csdnimg.cn/direct/cf1125f5ae8c40ecb61433aa140c42f0.png)![]()**编辑**

xtuner是什么?（rag结果）

![](https://i-blog.csdnimg.cn/direct/8eab5bc7516b4008824fa0864e096bed.png)![]()**编辑**

xtuner支持哪些模型

![](https://i-blog.csdnimg.cn/direct/5ef4e2d76e734457aeacf5a4af091954.png)![]()**编辑**
