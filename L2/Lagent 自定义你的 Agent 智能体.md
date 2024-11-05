Lagent 自定义你的 Agent 智能体
Lagent 介绍
Lagent 是一个轻量级开源智能体框架，旨在让用户可以高效地构建基于大语言模型的智能体。同时它也提供了一些典型工具以增强大语言模型的能力。

Lagent 目前已经支持了包括 AutoGPT、ReAct 等在内的多个经典智能体范式，也支持了如下工具：

Arxiv 搜索
Bing 地图
Google 学术搜索
Google 搜索
交互式 IPython 解释器
IPython 解释器
PPT
Python 解释器
其基本结构如下所示：
![alt text](image/image0.png)

环境配置
开发机选择 30% A100，镜像选择为 Cuda12.2-conda。

首先来为 Lagent 配置一个可用的环境。
```bash
# 创建环境
conda create -n agent_camp3 python=3.10 -y
# 激活环境
conda activate agent_camp3
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖包
pip install termcolor==2.4.0
pip install lmdeploy==0.5.2

# 创建目录以存放代码
mkdir -p /root/agent_camp3
# 安装agent
cd /root/agent_camp3
git clone https://github.com/InternLM/lagent.git
cd lagent && git checkout 81e7ace && pip install -e . && cd ..
pip install griffe==0.48.0

#启动lagent
conda activate agent_camp3
lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --model-name internlm2_5-7b-chat
```
调用api启动stream
```bash
cd /root/agent_camp3/lagent
conda activate agent_camp3
streamlit run examples/internlm2_agent_web_demo.py
```
报错
```bash
ImportError: cannot import name 'AutoRegister' from 'class_registry' (/root/.conda/envs/agent_camp3/lib/python3.10/site-packages/class_registry/__init__.py)
```
要再安装个依赖 
```bash
pip install class_registry
```
节约算力启用 internlm2-chat-1.8b

算力占用
![alt text](image/image-2.png)
修改 examples/internlm2_agent_web_demo.py 

调用模型api地址跟 lmdepoly启动后地址一致
![alt text](image/image.png)

启动：
![alt text](image/image-1.png)

![alt text](image/image-3.png)
![alt text](image/image00.png)

# 基于 Lagent 自定义智能体
在本节中，我们将带大家基于 Lagent 自定义自己的智能体。

Lagent 中关于工具部分的介绍文档位于 https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html 。

使用 Lagent 自定义工具主要分为以下几步：

## 继承 BaseAction 类
实现简单工具的 run 方法；或者实现工具包内每个子工具的功能
简单工具的 run 方法可选被 tool_api 装饰；工具包内每个子工具的功能都需要被 tool_api 装饰
下面我们将实现一个调用 MagicMaker API 以完成文生图的功能。

首先，我们先来创建工具文件：
```bash
cd /root/agent_camp3/lagent
touch lagent/actions/magicmaker.py
```
#### 然后，我们将下面的代码复制进入 
/root/agent_camp3/lagent/lagent/actions/magicmaker.py

```python
import json
import requests

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode


class MagicMaker(BaseAction):
    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
    
    @tool_api
    def generate_image(self, keywords: str) -> dict:
        """Run magicmaker and get the generated image according to the keywords.

        Args:
            keywords (:class:`str`): the keywords to generate image

        Returns:
            :class:`dict`: the generated image
                * image (str): path to the generated image
        """
        try:
            response = requests.post(
                url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
                data=json.dumps({
                    "official": True,
                    "prompt": keywords,
                    "style": self.style,
                    "poseT": False,
                    "aspectRatio": self.aspect_ratio
                }),
                headers={'content-type': 'application/json'}
            )
        except Exception as exc:
            return ActionReturn(
                errmsg=f'MagicMaker exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        image_url = response.json()['data']['imgUrl']
        return {'image': image_url}
```
#### 最后，我们修改 /root/agent_camp3/lagent/examples/internlm2_agent_web_demo.py 来适配我们的自定义工具。

在 from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter 的下一行添加 from lagent.actions.magicmaker import MagicMaker
在第27行添加 MagicMaker()。
```python
from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
+ from lagent.actions.magicmaker import MagicMaker
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol

...
        action_list = [
            ArxivSearch(),
+             MagicMaker(),
        ]
```
接下来，启动 Web Demo 来体验一下吧！我们同时启用两个工具，然后输入“请帮我生成一幅山水画”
工具调用成功，但是api 可能有点问题
https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate
![alt text](image/image-4.png)
![alt text](image/image-5.png)
![alt text](image/image-6.png)
搜索论文正常
![alt text](image/image-7.png)
