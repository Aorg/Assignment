 ![1729033516631](image/白嫖A100-git/1729033516631.png)

# 进入InterStudio

##### 这节课是为了让大家熟悉使用InterStudio平台，以便后续开发

InterStudio平台是算力平台，可以通过平台使用A100,还可以使用“书生”团队集成好的环境、工具，快速部署LLMs.

## 进入平台：

记得报名，获得免费算力，地址：[InternStudio](https://studio.intern-ai.org.cn/ "InternStudio")

![](https://i-blog.csdnimg.cn/direct/e19d2b2b7da54d9b890e0bdb9a50d728.png)![]()**编辑**

## 配置ssh：

![](https://i-blog.csdnimg.cn/direct/e2c167e0de874b57b995ea088719549f.png)![]()**编辑**

具体的配置ssh key方法 请看：

[https://aicarrier.feishu.cn/wiki/VLS7w5I22iQWmTk0ExpczIKcnpf](https://aicarrier.feishu.cn/wiki/VLS7w5I22iQWmTk0ExpczIKcnpf "https://aicarrier.feishu.cn/wiki/VLS7w5I22iQWmTk0ExpczIKcnpf")

## 创建开发机：

报好名进群登记后就有50点算力啦，直接创建一个最小资源的开发机熟悉一下，

cuda环境选和自己设备最接近的，以便于日后迁移。

![](https://i-blog.csdnimg.cn/direct/6aba29d216e242ee8486aec1b76028dd.png)![]()**编辑**

### 数据：

所有的数据都在“我的云盘”，可以理解为这就是你的系统盘，所以只要配好一套环境，新的开发及都会使用同一个conda。文件夹数据都会在相同的位置。

## 进入开发机：

![](https://i-blog.csdnimg.cn/direct/57b74e3186f642ff8a955487c9fd6d0b.png)![]()**编辑**

有两种选择，

### 本地使用Vscode连接服务器

点击ssh连接

![](https://i-blog.csdnimg.cn/direct/99c21f387c0f4e0b87eeae45e5fe353b.png)![]()**编辑**

复制登陆命令

![](https://i-blog.csdnimg.cn/direct/94028eece8d140d1a524140e8bbaf34d.png)![]()**编辑**

回车会进入ssh config 文件保存，因为已经配置ssh key不用管密码，

![](https://i-blog.csdnimg.cn/direct/780211d0a6ea42f38a2b081024064d8a.png)![]()**编辑**

在ssh列表中找到ssh.intern-ai.org.cn

![](https://i-blog.csdnimg.cn/direct/8f0c55bf71954aab942a8385ce9ac628.png)![]()**编辑**

点击直接连接，实现本地vscode连接interStudio服务器

![](https://i-blog.csdnimg.cn/direct/ea27a8e44df54e5bb977bec419a9cf27.png)![]()**编辑**

### 网页访问平台的Vscode,终端

点击进入开发机，有vscode和terminal选项

## 端口映射：

### 1.安装gradio

```bash
pip install gradio==4.29.0
```

![]()

###### 如果库冲突，按照指示修改

我需要修改的库版本如下:

* pip install importlib-metadata==6.6
* pip install requests~=2.29
* pip install urllib3~=2.0

### 2. 复制代码：

    创建一个hello.py文

```python
import socket
import re
import gradio as gr
 
# 获取主机名
def get_hostname():
    hostname = socket.gethostname()
    match = re.search(r'-(\d+)$', hostname)
    name = match.group(1)
  
    return name
 
# 创建 Gradio 界面
with gr.Blocks(gr.themes.Soft()) as demo:
    html_code = f"""
            <p align="center">
            <a href="https://intern-ai.org.cn/home">
                <img src="https://intern-ai.org.cn/assets/headerLogo-4ea34f23.svg" alt="Logo" width="20%" style="border-radius: 5px;">
            </a>
            </p>
            <h1 style="text-align: center;">☁️ Welcome {get_hostname()} user, welcome to the ShuSheng LLM Practical Camp Course!</h1>
            <h2 style="text-align: center;">😀 Let’s go on a journey through ShuSheng Island together.</h2>
            <p align="center">
                <a href="https://github.com/InternLM/Tutorial/blob/camp3">
                    <img src="https://oss.lingkongstudy.com.cn/blog/202406301604074.jpg" alt="Logo" width="20%" style="border-radius: 5px;">
                </a>
            </p>

            """
    gr.Markdown(html_code)

demo.launch()
```

![]()

    本地网页打开远程服务器运行的代码。

### 3.打开cmd.使用ssh连接服务器

![](https://i-blog.csdnimg.cn/direct/7b52098a11fe4a399d63ad30da71f17c.png)![]()**编辑**

### 4.运行代码：

打开网页

如果没有cmd连接ssh是无法打开的噢

![](https://i-blog.csdnimg.cn/direct/aeb5c841d4334e659281b26e01910b33.png)![]()**编辑**

## linux常用命令

建议大家查看活动的官方教程[Docs](https://aicarrier.feishu.cn/wiki/XZChwwDsciyFyHk5mGTc1EKinkc "Docs")

其中比较重要的是查看自己的GPU

1.studio-smi

2.克隆环境

```bash
tar --skip-old-files -xzvf /share/pkgs.tar.gz -C ${CONDA_HOME}

conda create -n $target --clone ${SHARE_CONDA_HOME}/${source}
```

![]()

第一步，将新的conda环境创建到/share/conda_envs下

> conda create -p /share/conda_envs/xxx python=3.1x

第二步，将本机/root/.conda/pkgs下的文件拷贝到/share/pkgs中，重新压缩并替换(此步骤是为了把conda创建过程中大的公共包存储起来，避免重复下载)

> cp -r -n /root/.conda/pkgs/* /share/pkgs/
>
> cd /share && tar -zcvf pkgs.tar.gz pkgs

第三步，更新install_conda_env.sh中的list函数，增加新的conda环境说明。

## 常见问题

1. ### InternStudio 开发机的环境玩坏了，如何初始化开发机环境

**慎重执行！！！！所有数据将会丢失，仅限 InternStudio 平台，自己的机器千万别这么操作**

* 第一步本地终端 ssh 连上开发机（一定要 ssh 连接上操作，不能在 web 里面操作！！！）
* 第二步执行 `rm -rf /root`，大概会等待10分钟
* 第三步重启开发机,系统会重置 /root 路径下的配置文件
* 第四步 `ln -s /share /root/share`
