扫码参加-白嫖A100

![1729033461435](image/提示词工程课程/1729033461435.png)

![img](https://i-blog.csdnimg.cn/direct/3af9e40cd47d4207a8ef0ab78480b986.png)![]()**编辑**

### 内容来源：[Tutorial/opencompass/readme.md at camp2 · InternLM/Tutorial · GitHub](https://github.com/InternLM/Tutorial/blob/camp2/opencompass/readme.md "Tutorial/opencompass/readme.md at camp2 · InternLM/Tutorial · GitHub")

### 概览

在 OpenCompass 中评估一个模型通常包括以下几个阶段：配置 -> 推理 -> 评估 -> 可视化。

* 配置：这是整个工作流的起点。您需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。
* 推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。如果需要了解该问题及解决方案，可以参考 FAQ: 效率。
* 可视化：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。你也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。 接下来，我们将展示 OpenCompass 的基础用法，展示书生浦语在 `C-Eval` 基准任务上的评估。它们的配置文件可以在 `configs/eval_demo.py` 中找到。

# 实战：

## github下载，并按照包：

```bash
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .

apt-get update
apt-get install cmake
pip install -r requirements.txt
pip install protobuf
```

![]()

## 复制测评数据集：

```bash
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip
```

![]()

## 查询可测评数据集：

```bash
python tools/list_configs.py internlm ceval
```

![]()

 类似于linux命令中的grep，在list_configs.py后面输入关键字。如果想查别的模型可以输入模型名称：

![](https://i-blog.csdnimg.cn/direct/e06e28310f334f7e8e534a133cacb38f.png)![]()**编辑**

想查别的数据集可以直接输入数据集名称：

![](https://i-blog.csdnimg.cn/direct/63d52ab2cbe14ff4b3f80342b4efe4ac.png)![]()**编辑**

## 根据配置信息，开始测评：

![](https://i-blog.csdnimg.cn/direct/805d057a2c2346f0bb16e5773ec7f3e7.png)![]()**编辑**

看到：** [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...** 就成功了

使用到6G左右的显存：

![](https://i-blog.csdnimg.cn/direct/78a6b2f30fcd4000b4c51e97a9572b85.png)![]()**编辑**

## 测评结果：

terminal会显示，也会存储到指定文件夹下，并将位置打印出来

![](https://i-blog.csdnimg.cn/direct/856d462d85664741b580864e1d38b015.png)![]()**编辑**

![](https://i-blog.csdnimg.cn/direct/01ea35911a3b47fcaf382979da7bdf73.png)![]()**编辑**

# 进阶任务：

## 1.主观评价：

教程：

[https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/subjective_evaluation.html](https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/subjective_evaluation.html "https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/subjective_evaluation.html")

### 下载数据集 AlignBench

修改数据格式：

> {
>
> "question": "高音单簧管和高音萨克斯的调性相同吗？如果相同，请说出他们的调性，如果不同，请分别说出他们的调性",
>
> "capability": "专业能力",
>
> "others": {
>
> "subcategory": "音乐",
>
> "reference": "高音单簧管和高音萨克斯的调性不同。高音单簧管的调性通常为E♭，而高音萨克斯的调性则为B♭。",
>
> "question_id": 1
>
> }
>
> },

### 申请了火星大模型免费API，作为judge_models

[控制台-讯飞开放平台](https://console.xfyun.cn/services/bm35 "控制台-讯飞开放平台")

### 修改配置文件：

/root/opencompass/configs/datasets/subjective/alignbench/alignbench_judgeby_critiquellm.py

/root/opencompass/configs/eval_subjective_alignbench.py

```python
from mmengine.config import read_base
from opencompass.models.xunfei_api import XunFei
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners.sub_size import SubjectiveSizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks.subjective_eval import SubjectiveEvalTask
from opencompass.summarizers import AlignmentBenchSummarizer

with read_base():
    from .datasets.subjective.alignbench.alignbench_judgeby_critiquellm import subjective_datasets
    from .datasets.subjective.alpaca_eval.alpacav2_judgeby_gpt4 import subjective_datasets as alpacav2
    from .models.qwen.hf_qwen_7b import models
    from .summarizers.subjective import summarizer
    from.models.hf_internlm.hf_internlm2_1_8b import models



api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ]
)

# -------------Inference Stage ----------------------------------------
# For subjective evaluation, we often set do sample for models

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='internlm2-chat-1.8b-hf',
        path="internlm/internlm2-chat-1_8b",
        tokenizer_path='internlm/internlm2-chat-1_8b',
        model_kwargs=dict(
            trust_remote_code=True,
            device_map='auto',
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
            trust_remote_code=True,
        ),
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8,
        meta_template=api_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
        # generation_kwargs = {"eos_token_id": [2, 92542]},
        generation_kwargs=dict(
             do_sample=True,
         ),
    )
]
# models = [
#     dict(
#         type=HuggingFaceChatGLM3,
#         abbr='chatglm3-6b-hf',
#         path='THUDM/chatglm3-6b',
#         tokenizer_path='THUDM/chatglm3-6b',
#         model_kwargs=dict(
#             device_map='auto',
#             trust_remote_code=True,
#         ),
#         tokenizer_kwargs=dict(
#             padding_side='left',
#             truncation_side='left',
#             trust_remote_code=True,
#         ),
#         generation_kwargs=dict(
#             do_sample=True,
#         ),
#         meta_template=api_meta_template,
#         max_out_len=2048,
#         max_seq_len=4096,
#         batch_size=8,
#         run_cfg=dict(num_gpus=1, num_procs=1),
#     )
# ]

datasets = [*subjective_datasets]

# -------------Evalation Stage ----------------------------------------

## ------------- JudgeLLM Configuration
judge_models = [
#     dict(
#     abbr='GPT4-Turbo',
#     type=OpenAI,
#     path='gpt-4-1106-preview',
#     key='xxxx',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
#     meta_template=api_meta_template,
#     query_per_second=16,
#     max_out_len=2048,
#     max_seq_len=2048,
#     batch_size=8,
#     temperature=0,
# )
    dict(
            abbr='Spark-v3-5',
            type=XunFei,
            appid="xxxxxxxxxxxxxxx",
            path='ws(s)://spark-api.xf-yun.com/v3.5/chat',
            api_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxx",
            api_key = "xxxxxxxxxxxxxxxxxxxxxx",
            query_per_second=1,
            max_out_len=2048,
            max_seq_len=2048,
            batch_size=8),
]

## ------------- Evaluation Configuration
eval = dict(
    partitioner=dict(
        type=SubjectiveSizePartitioner, max_task_size=1000, mode='singlescore', models=models, judge_models=judge_models,
    ),
    runner=dict(type=LocalRunner, max_num_workers=2, task=dict(type=SubjectiveEvalTask)),
)

summarizer = dict(type=AlignmentBenchSummarizer, judge_type='general')

work_dir = 'outputs/alignment_bench/'
```

![]()

### 结果：

![](https://i-blog.csdnimg.cn/direct/7590ca77108f47b58f4ee1ae27e554f5.png)![]()**编辑**

### output/config

```python
alpacav2=[
    dict(abbr='alpaca_eval',
        eval_cfg=dict(
            evaluator=dict(
                prompt_template=dict(
                    template=dict(
                        begin=[
                            dict(fallback_role='HUMAN',
                                prompt='You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers.',
                                role='SYSTEM'),
                            ],
                        round=[
                            dict(prompt='\nI require a leaderboard for various large language models. I\'ll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.\n\n## Instruction\n\n{\n    "instruction": "{question}",\n}\n\n## Model Outputs\n\nHere are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.\n\n{\n    {\n        "model_identifier": "m",\n        "output": "{prediction}"\n    },\n    {\n        "model_identifier": "M",\n        "output": "{prediction2}"\n    }\n}\n\n## Task\n\nEvaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.\n\n## Best Model Identifier\n',
                                role='HUMAN'),
                            ]),
                    type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
                type='opencompass.openicl.icl_evaluator.LMEvaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=4096,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{question}',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='alpaca_eval',
        path='./data/subjective/alpaca_eval',
        reader_cfg=dict(
            input_columns=[
                'question',
                ],
            output_column='judge'),
        type='opencompass.datasets.SubjectiveCmpDataset'),
    ]
api_meta_template=dict(
    round=[
        dict(api_role='HUMAN',
            role='HUMAN'),
        dict(api_role='BOT',
            generate=True,
            role='BOT'),
        ])
datasets=[
    dict(abbr='alignment_bench',
        alignment_bench_config_name='multi-dimension',
        alignment_bench_config_path='/root/opencompass/AlignBench/config',
        eval_cfg=dict(
            evaluator=dict(
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt='{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n',
                                role='HUMAN'),
                            ]),
                    type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
                type='opencompass.openicl.icl_evaluator.LMEvaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=2048,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{question}',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='alignment_bench',
        path='/root/opencompass/AlignBench/data',
        reader_cfg=dict(
            input_columns=[
                'question',
                'capability',
                'critiquellm_prefix',
                ],
            output_column='judge'),
        type='opencompass.datasets.AlignmentBenchDataset'),
    ]
eval=dict(
    partitioner=dict(
        judge_models=[
            dict(abbr='Spark-v3-5',
                api_key='xxxxxxxxxxxxxxxxxx',
                api_secret='xxxxxxxxxxxxxxxxx',
                appid='xxxxxxxxxxx',
                batch_size=8,
                max_out_len=2048,
                max_seq_len=2048,
                path='ws(s)://spark-api.xf-yun.com/v3.5/chat',
                query_per_second=1,
                type='opencompass.models.xunfei_api.XunFei'),
            ],
        max_task_size=1000,
        mode='singlescore',
        models=[
            dict(abbr='internlm2-chat-1.8b-hf',
                batch_size=8,
                generation_kwargs=dict(
                    do_sample=True),
                max_out_len=2048,
                max_seq_len=4096,
                meta_template=dict(
                    round=[
                        dict(api_role='HUMAN',
                            role='HUMAN'),
                        dict(api_role='BOT',
                            generate=True,
                            role='BOT'),
                        ]),
                model_kwargs=dict(
                    device_map='auto',
                    trust_remote_code=True),
                path='internlm/internlm2-chat-1_8b',
                run_cfg=dict(
                    num_gpus=1,
                    num_procs=1),
                tokenizer_kwargs=dict(
                    padding_side='left',
                    truncation_side='left',
                    trust_remote_code=True,
                    use_fast=False),
                tokenizer_path='internlm/internlm2-chat-1_8b',
                type='opencompass.models.HuggingFaceCausalLM'),
            ],
        type='opencompass.partitioners.sub_size.SubjectiveSizePartitioner'),
    runner=dict(
        max_num_workers=2,
        task=dict(
            type='opencompass.tasks.subjective_eval.SubjectiveEvalTask'),
        type='opencompass.runners.LocalRunner'))
judge_models=[
    dict(abbr='Spark-v3-5',
        api_key='9c7849c19377c2748db367a289458fbe',
        api_secret='MTgyZGYyNjI3OWE3MzlhMzE4MmQ4N2Jh',
        appid='bec61d4e',
        batch_size=8,
        max_out_len=2048,
        max_seq_len=2048,
        path='ws(s)://spark-api.xf-yun.com/v3.5/chat',
        query_per_second=1,
        type='opencompass.models.xunfei_api.XunFei'),
    ]
models=[
    dict(abbr='internlm2-chat-1.8b-hf',
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True),
        max_out_len=2048,
        max_seq_len=4096,
        meta_template=dict(
            round=[
                dict(api_role='HUMAN',
                    role='HUMAN'),
                dict(api_role='BOT',
                    generate=True,
                    role='BOT'),
                ]),
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True),
        path='internlm/internlm2-chat-1_8b',
        run_cfg=dict(
            num_gpus=1,
            num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False),
        tokenizer_path='internlm/internlm2-chat-1_8b',
        type='opencompass.models.HuggingFaceCausalLM'),
    ]
subjective_datasets=[
    dict(abbr='alignment_bench',
        alignment_bench_config_name='multi-dimension',
        alignment_bench_config_path='/root/opencompass/AlignBench/config',
        eval_cfg=dict(
            evaluator=dict(
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt='{critiquellm_prefix}[助手的答案开始]\n{prediction}\n[助手的答案结束]\n',
                                role='HUMAN'),
                            ]),
                    type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
                type='opencompass.openicl.icl_evaluator.LMEvaluator'),
            pred_role='BOT'),
        infer_cfg=dict(
            inferencer=dict(
                max_out_len=2048,
                type='opencompass.openicl.icl_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt='{question}',
                            role='HUMAN'),
                        ]),
                type='opencompass.openicl.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                type='opencompass.openicl.icl_retriever.ZeroRetriever')),
        name='alignment_bench',
        path='/root/opencompass/AlignBench/data',
        reader_cfg=dict(
            input_columns=[
                'question',
                'capability',
                'critiquellm_prefix',
                ],
            output_column='judge'),
        type='opencompass.datasets.AlignmentBenchDataset'),
    ]
summarizer=dict(
    judge_type='autoj',
    type='opencompass.summarizers.AlignmentBenchSummarizer')
work_dir='outputs/alignment_bench/20240926_062247'
```

![]()

log:

/root/opencompass/outputs/alignment_bench/20240926_062247/logs/eval/internlm2-chat-1.8b-hf/alignment_bench_13.out

> /bin/sh: 1: 0: not found

### 找不到BUG在哪...

## 2.使用 OpenCompass 评测 InternLM2-Chat-1.8B 模型使用 LMDeploy部署后在 ceval 数据集上的性能（选做）
