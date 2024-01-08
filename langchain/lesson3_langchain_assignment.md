# **基础作业**：

## 复现课程知识库助手搭建过程 (截图)
### 1.环境代码准备
    配置环境 下载模型 下载代码
![image.png](images/3langchain/download.jpg)
### 2.化为向量知识库 持久化储存
运行 create_db.py
![image.png](images/3langchain/1.jpg)

### 3.基于InternLM模型自定义 LLM 类
    model_path = "下载的internlm-chat-7b模型地址"
### 4.使用知识库助手
    persist_directory=持久化知识库位置
     HuggingFaceEmbeddings(model_name=下载的向量模型位置)
     llm = InternLM_LLM(model_path = 下载的internlm-chat-7b模型地址)
    运行 run_gradio.py
![image.png](images/3langchain/3.jpg)





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



