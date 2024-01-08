# 导入必要的库
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
rag_model = "/home/chy/api/tutorial/langchain/demo/model/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
llm_model = "/home/chy/.cache/modelscope/hub/Shanghai_AI_Laboratory"
def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=rag_model)

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    llm = InternLM_LLM(model_path = llm_model)

    template = """
                    按照给定的格式回答以下问题。
                    回答时需要遵循以下用---括起来的格式：

                    ---
                    Question: 需要回答的问题
                    Thought: 回答Question我需要做些什么，切入点是什么，思维链路步骤是什么。
                    record: 记住想到的步骤，一步一步回答。 
                    answer: 回答record的每一步记录的问题和步骤。
                    Observation: 回看所有步骤，验证并回答问题，由于面向学生年龄段低，水平参差不齐，请将每一步都详细的解答。
                    Thought: 我现在知道最终答案。如果不太确定，可以重复多次Thought,record,answer,Observation
                    ...（这个思考/行动/行动输入/观察可以重复N次）
                    Final Answer: 原始输入问题的最终答案
                    ---
                    现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。                     
                    使用以下上下文来回答用户的问题。如果你不知道答案，则重复Thought步骤。
                    Question: {question}
                    可参考的上下文：
                    ···
                    {context}
                    ···
                    如果给定的上下文无法让你做出回答，则重复Thought步骤。总是使用中文回答。
                    验证过的回答:
                    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)

    # 运行 chain
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

class Model_center():
    """
    存储问答 Chain 的对象 
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)
        # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        
    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
