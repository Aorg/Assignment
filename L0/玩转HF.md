# 启动Codespaces
是Github推出的线上代码平台，提供了一系列templates，我们这里选择Jupyter Notebook进行创建环境。创建好环境后，可以进入网页版VSCode的界面，这就是CodeSpace提供给我们的在线编程环境。

codespaces是一个免费代码编译空间，已经帮助大家集成了相应编译工具

是不用翻墙访问huggingface
## 安装transformers
```bash
pip install transformers==4.38
pip install sentencepiece==0.1.99
pip install einops==0.8.0
pip install protobuf==5.27.2
pip install accelerate==0.33.0
```

### 实际可用空间是workspaces 32G

![alt text](image-3.png)
![alt text](image-1.png)
和vscode一样的界面
![alt text](image-2.png)

# 执行部分文件donwload
示例代码
```python
import os
from huggingface_hub import hf_hub_download

# 指定模型标识符
repo_id = "internlm/internlm2_5-7b"

# 指定要下载的文件列表
files_to_download = [
    {"filename": "config.json"},
    {"filename": "model.safetensors.index.json"}
]

# 创建一个目录来存放下载的文件
local_dir = f"{repo_id.split('/')[1]}"
os.makedirs(local_dir, exist_ok=True)

# 遍历文件列表并下载每个文件
for file_info in files_to_download:
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_info["filename"],
        local_dir=local_dir
    )
    print(f"{file_info['filename']} file downloaded to: {file_path}")
    
```
### 运行结果

![alt text](image.png)

## 执行interlm2.5-1.8b下载 运行

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-1_8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-1_8b", torch_dtype=torch.float16, trust_remote_code=True)
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
# 肯定得注释得
output = model.generate(**inputs, **gen_kwargs)
output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
print(output)
```
按照示例代码运行
下载很快
![alt text](image-4.png)
本地下载
修改部分代码
```python
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
```

下载模型会存放在下载文件当下位置

![alt text](image-5.png)
### cpu运行回答
![alt text](image-10.png)
![alt text](image-20.png)

# huggingface上传
## huggingface spaces上传
[\[进入https://huggingface.co/new-space\]](https://huggingface.co/new-space)

创建新的空间选 static
![alt text](image-7.png)
### 空间性能和价格
![alt text](image-8.png)

## 在空间按中上传自己的代码
进入后啥都没有需要上传代码

进入空间 点击右上角，clone 
![alt text](image-9.png)
![alt text](image-6.png)

### clone 到自己编译空间
填写代码
教程修改了 index.html
```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width" />
  <title>My static Space</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
    }
    body {
      display: flex;
      justify-content: center;
      align-items: center;
    }
    iframe {
      width: 430px;
      height: 932px;
      border: none;
    }
  </style>
</head>
<body>
  <iframe src="https://colearn.intern-ai.org.cn/cobuild" title="description"></iframe>
</body>
</html>

```
### 再更新huggingface space代码
#### 获取huggingface可写入Access Tokens
回到huggingface
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)
copy 好再关闭，为了安全 就这一次机会看见这个独一无二的accesse tokens， 以后就只能刷新或者新建accesse tokens了
### git 写入权限
bash模板
```bash
git remote set-url origin https://<user_name>:<token>@huggingface.co/<repo_path>
```
<user_name>替换huggingface 用户名

<token>替换为Access Tokens（可写入的）

<repo_path>替换为你的仓库名，如下 可用直接copy
![alt text](image-14.png) 
```bash
git remote set-url origin https://Ruiy:hf_xxxxxxxxxxxxxxxxxxxxxxkSE@huggingface.co/Ruiy/Aorg
```

```bash 
git add .
git commit -m "update: colearn page"
git push
```
### 完成作业 可访问地址：
https://huggingface.co/spaces/Ruiy/Aorg
按照官方教程改的html 可用看到如下界面
![alt text](image-15.png)

## huggingface模型上传
模型文件较大

通过CLI上传 
### Hugging Face跟Git相关联
通常大模型的模型文件都比较大，因此我们需要安装git lfs，对大文件系统支持。

在自己的环境下安装
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# sudo apt-get install git-lfs # CodeSpace里面可能会有aptkey冲突且没有足够权限
git lfs install # 直接在git环境下配置git LFS
pip install huggingface_hub
```
![alt text](image-16.png)

```bash
git config --global credential.helper store
huggingface-cli login
```
```
# 输入你的huggingface assess token (可写入的)
Enter your token (input will not be visible): 
```
![alt text](image-17.png)

如果之前已经登陆过的 需要更换
使用下面命令 清楚之前的token
```bash
git config --global --unset credential.helpergit config --global credential.helper cache
git credential-cache exit
```
现在能将模型顺利上传huggingface了
在huggingface 新建模型库
### +huggingface 创建自己的模型
将库传到自己的服务器，再上传模型文件
![alt text](image-18.png)
![alt text](image-19.png)
![alt text](image-21.png)


只上传我们刚刚下载好的config.json，把它复制粘贴进这个文件夹里面，还可以写一个README.md文件，比如可以粘贴以下内容：

```
- hugging face模型上传测试
- 更多内容请访问 https://github.com/InternLM/Tutorial/tree/camp4
```
现在可以用git提交到远程仓库
```bash
cd intern_study_L0_4
git add .
git commit -m "add:intern_study_L0_4"
git push
```
权限有问题再来一遍
```bash
git remote set-url origin https://Ruiy:hf_xxxxxxxxxxxxxxxxxxxxxxkSE@huggingface.co/Ruiy/internlm2_5-1_8b
```
### 完成作业 可访问地址：
https://huggingface.co/Ruiy/internlm2_5-1_8b
![alt text](image-22.png)
![alt text](image-23.png)


# 2.2 魔搭社区平台
ModelScope 是一个“模型即服务”(MaaS)平台，由阿里巴巴集团的达摩院推出和维护。它旨在汇集来自AI社区的最先进的机器学习模型，并简化在实际应用中使用AI模型的流程。通过ModelScope，用户可以轻松地探索、推理、微调和部署各种AI模型。

注册登录ModelScope平台，进入导航栏模型库，可以搜索internlm找到相关模型（但是这也包括非官方模型），在这里我们可以搜索 internlm2_5-chat-1_8b，下载1.8b的对话模型，也类似于hugging face 一样拥有具体的详情页。

### 2.2.1 创建开发机
我们选择 10% 的开发机，镜像选择为 Cuda-12.2。在输入开发机名称后，点击创建开发机。

image

创建好开发机后，进入开发机

image

接着在当前终端上可以输入命令了，这里可以直接粘贴以下命令。最好一行一行粘贴等每个命令跑完之后再粘贴下一行

### 2.2.2 环境配置
为ms_demo创建一个可用的conda虚拟环境，可以和其他环境区分开来

#### 激活环境
conda activate /root/share/pre_envs/pytorch2.1.2cu12.1

#### 安装 modelscope
pip install modelscope -t /root/env/maas
pip install numpy==1.26.0  -t /root/env/maas
pip install packaging -t /root/env/maas
注意：为了保证能够找到我们每次装的依赖，每次新建一个终端之后都需要导出path 如果不改变终端，导出一次就够了

export PATH=$PATH:/root/env/maas/bin
export PYTHONPATH=/root/env/maas:$PYTHONPATH
接着创建我们的demo目录

mkdir -p /root/ms_demo
### 2.2.3 下载指定多个文件
internlm2_5-7b-chat 考虑到7B的模型文件比较大，这里我们先采用modelscope的cli工具（当然hugging face也有）来下载指定文件，在命令行输入以下命令
modelscope download \
    --model 'Shanghai_AI_Laboratory/internlm2_5-7b-chat' \
    tokenizer.json config.json model.safetensors.index.json \
    --local_dir '/root/ms_demo'
刷新一下文件目录，就能看到在ms_demo中下载了指定的json文件。

internlm2_5-1_8b-chat
modelscope download \
    --model 'Shanghai_AI_Laboratory/internlm2_5-1_8b-chat' \
    tokenizer.json config.json model.safetensors.index.json \
    --local_dir '/root/ms_demo'
### 2.2.4 上传模型
魔搭社区
类似HF，也有一套创建模型的界面。不同的是，它具有审核机制，当符合它的社区规范时才会被公开。那么当上传正常的模型文件后，审核一般就会通过了。

上传文件的方法可以直接通过平台添加文件，也可以通过git下载模型后进行修改和上传文件

#Git模型下载
git clone https://www.modelscope.cn/<your_username>/<your_model>
# 2.3 魔乐社区平台
魔乐社区（Modelers）是一个提供多样化、开源模型的平台，旨在促进开发者和研究人员在最先进的模型和流行应用上进行协作。

### 2.3.1 下载internlm2_5-chat-1_8b模型
这里我们可以继续使用我们刚刚创建的InterStudio开发机

cd /
mkdir ml_demo
cd ml_demo
然后我们可以下载该模型，这里

### 确保安装git-lfs 保证大文件的正常下载
apt-get install git-lfs
git lfs install
### clone 仓库
git clone https://modelers.cn/Intern/internlm2_5-1_8b-chat.git
刷新一下文件夹，即可在ml_demo中找到下载好的模型文件，在魔乐社区中，还推荐了一个新的深度学习开发套件openMind Library，除了常用的Transforms的API，也可以探索如何使用openMind来加载模型

### 确保按指南安装好openmind后
from openmind import AutoModel
model = AutoModel.from_pretrained("Intern/internlm2_5-1_8b-chat", trust_remote_code=True")
openMind Library是一个深度学习开发套件，通过简单易用的API支持模型预训练、微调、推理等流程。 openMind Library通过一套接口兼容PyTorch和MindSpore等主流框架，同时原生支持昇腾NPU处理器。

### 2.3.2 上传模型
在魔乐社区一般有两种方法，第一种是安装好openmid后使用openmind的API来上传文件，另一个就是用git命令来推送文件，跟一般的git工作流相类似。可参考上传文件 | 魔乐社区

# 额外的
codespace 更改代码 更新至仓库
总是遇到问题
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
存放位置
```bash
cat /home/codespace/.ssh/id_rsa.pub
```
位置不对 每次上传
ssh key都不对
![alt text](image-24.png)
执行
```bash
mkdir -p ~/.ssh
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
```
这样就能正确更新了