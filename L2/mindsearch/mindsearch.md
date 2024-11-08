# Mindsearch
利用硅基流动免费api调用大模型，通过github codespace，上传至huggingface(git codespace 传至huggingface不需要梯子)
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
![alt text](image-7.png)
![alt text](image-5.png)
![alt text](image-9.png)
报错有可能包冲突
字体设置看这里：
https://blog.csdn.net/Aorg1/article/details/138609956

# 部署到huggingface
将 MindSearch 部署到 HuggingFace Space。
打开 https://huggingface.co/spaces ，
并点击 Create new Space，如下图所示
![alt text](image-8.png)
点击setting
![alt text](image-10.png)

本地创建目录
```bash
mkdir -p /root/mindsearch/mindsearch_deploy
# 准备复制文件
cd /root/mindsearch
cp -r /root/mindsearch/MindSearch/mindsearch /root/mindsearch/mindsearch_deploy
cp /root/mindsearch/MindSearch/requirements.txt /root/mindsearch/mindsearch_deploy
# 创建 app.py 作为程序入口
touch /root/mindsearch/mindsearch_deploy/app.py
```
```bash
git clone https://huggingface.co/spaces/<repo_path>
```
![alt text](image-11.png)
我的会生成test_mindsearch文件
```bash
-cp -r /workspace/mindsearch/MindSearch/mindsearch /workspace/mindsearch/test_mindsearch
-cp /workspace/mindsearch/MindSearch/mindsearch/requirements.txt /workspace/mindsearch/test_mindsearch
git add .
git commit -m 'mindsearch'
git remote set-url origin https://Ruiy:<token>@huggingface.co/spaces/Ruiy/test_mindsearch
git push
```
token 点击头像选择acess token ，write 权限复制token
已上传至huggingface 
回到huggingface页面，点击app
开始自动安装
![alt text](image-12.png)


