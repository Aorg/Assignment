# Mindsearch
利用硅基流动免费api调用大模型，codespace跑通
## 硅基流动 API Keyapi
<https://cloud.siliconflow.cn/account/ak>
注册
![alt text](image-3.png)
一会复制key配置环境中

## codespaces创建虚拟机
![alt text](image.png)
进入后
选择python环境类似vscode
![alt text](image-1.png)
## 安装环境
```bash
#copy mindsearch代码库
mkdir -p /workspaces/mindsearch
cd /workspaces/mindsearch
git clone https://github.com/InternLM/MindSearch.git
cd MindSearch && git checkout b832275 && cd ..
#用codespaces免去隔离环境
# 安装依赖
pip install -r /workspaces/mindsearch/MindSearch/requirements.txt
# 以下两个包和gradio==5.3.0冲突，gradio暂时不安装
```
## mindSearch后端启动
```bash
export SILICON_API_KEY=硅基流动API_Keyapi(第一步)
cd /workspaces/mindsearch/MindSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch
```
![alt text](image-4.png)
## 前端启动
```bash
conda activate mindsearch
cd /workspaces/mindsearch/MindSearch
python frontend/mindsearch_gradio.py
```
