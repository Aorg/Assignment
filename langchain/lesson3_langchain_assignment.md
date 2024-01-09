# **基础作业**：

## 复现课程知识库助手搭建过程 (截图)
### 1.环境代码准备
    配置环境 下载模型 下载代码
![image.png](images/download.jpg)
### 2.化为向量知识库 持久化储存
运行 create_db.py
![image.png](images/1.jpg)

### 3.基于InternLM模型自定义 LLM 类
    model_path = "下载的internlm-chat-7b模型地址"
### 4.使用知识库助手
    persist_directory=持久化知识库位置
     HuggingFaceEmbeddings(model_name=下载的向量模型位置)
     llm = InternLM_LLM(model_path = 下载的internlm-chat-7b模型地址)
    运行 run_gradio.py
![image.png](images/3.jpg)





# **进阶作业**：
## 选择一个垂直领域，收集该领域的专业资料构建专业知识库，并搭建专业问答助手，并在 [OpenXLab](https://openxlab.org.cn/apps) 上成功部署（截图，并提供应用地址）
## 1环境配置
除了教程的环境，如果需要读取pdf,docx文件需要安装以下包
```shell
pip install pypdf
pip install pdfminer.six
pip install opencv-python
pip install pytesseract
pip install python-docx
```
## 2向量模型下载
创建 download_hf.py 下载向量模型
```python
import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /home/chy/api/tutorial/langchain/demo/model/sentence-transformer')
```
在bash中运行
```shell
nohup python download_hf.py > log_download_hf.log 2>&1 &
```
## 3.上传知识库
上传垂直领域知识库，create_db.py中修改变量
```python
#检索模型路径
model_name="model/sentence-transformer"
# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'
# 目标文件位置
tar_dir = [
    "文档位置1"，
    "文档位置2"
    ...
]
```
运行 create_db.py

目标位置检索文档->文档->向量知识库->持久化
## 3 InternLM 接入 LangChain
以下使用教程中的描述：

为便捷构建 LLM 应用，我们需要基于本地部署的 InternLM，继承 LangChain 的 LLM 类自定义一个 InternLM LLM 子类，从而实现将 InternLM 接入到 LangChain 框架中。完成 LangChain 的自定义 LLM 子类之后，可以以完全一致的方式调用 LangChain 的接口，而无需考虑底层模型调用的不一致。

基于本地部署的 InternLM 自定义 LLM 类并不复杂，我们只需从 LangChain.llms.base.LLM 类继承一个子类，并重写构造函数与 `_call` 函数即可：

```python
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """
        
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"
```

在上述类定义中，我们分别重写了构造函数和 `_call` 函数：对于构造函数，我们在对象实例化的一开始加载本地部署的 InternLM 模型，从而避免每一次调用都需要重新加载模型带来的时间过长；`_call` 函数是 LLM 类的核心函数，LangChain 会调用该函数来调用 LLM，在该函数中，我们调用已实例化模型的 chat 方法，从而实现对模型的调用并返回调用结果。

在整体项目中，我们将上述代码封装为 LLM.py，后续将直接从该文件中引入自定义的 LLM 类。

## 运行run_gradio.py
![image.png](images/5.jpg)


