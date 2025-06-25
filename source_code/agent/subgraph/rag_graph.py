import os
from typing import List, Literal

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from pydantic import BaseModel, Field

from source_code.agent.configuration import Configuration
from source_code.agent.llm import bge_embeddings, qwen_llm_7b, qwen_25_vl, bge_reranker
from source_code.agent.state import RAGState
from source_code.api.config import vector_dir, pdf_dir
from source_code.db.mongo import get_db


class OptimizedQuery(BaseModel):
    """
    优化后的查询结构
    """

    expanded_terms: List[str] = Field(description="相关术语、核心概念和关键词列表")


class ExpandCaption(BaseModel):
    caption: List[str] = Field(description="文本中包含的图表标题")


class ResponseDiscriminator(BaseModel):
    is_valid: bool = Field(description="回答是否有效")
    reason: str = Field(description="判断依据")


# 定义查询优化的提示模板
query_optimization_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个专业的查询优化助手。你的任务是从用户的相关问题中提取核心概念和关键词用于后续向量查询。
            请遵循以下规则：
            1. 提取核心概念和关键词
            2. 去除无关的修饰词
            3. 保持专业性和准确性，不要改变用户问题的原意

            输出格式要求：
            使用OptimizedQuery结构体，包含：
            - expanded_terms: 核心概念和关键词列表
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

# 定义图表标题提取的提示模版
expand_caption_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个专业的文本分析助手，你的任务是从文本中提取所有提到的图表标题。
            1.任务描述
                从用户输入的文本中识别并提取所有图表的标题或引用，包括但不限于"图X"、"表X"、"图X-Y"、"表X-Y"等格式。
            2.提取规则
                - 提取所有包含"图"或"表"字样后跟数字的内容
                - 提取时保留完整的图表标识，包括"图"或"表"字样和所有相关数字
                - 忽略图表标题后的描述性文本
                - 如果同一图表在文本中多次出现，只需提取一次
                - 严格按照原文中的格式提取
            3.示例
                输入：
                "图 1-5 表示了一个典型的中期染色体结构，主要包括短臂、长臂和着丝粒(centromere)。称为衍生染色体，可记述为der(2)和der(5)(见表1-2)。"
                输出：
                ["图 1-5", "表1-2"]
            4.输出格式
                请使用ExpandCaption格式化输出,包含：
                - caption：文本中包含的图表标题
            5.注意事项
                如果无法从文本中提取图表标题，则使用ExpandCaption输出空列表。
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)

query_optimization_model = qwen_llm_7b.with_structured_output(
    OptimizedQuery, method="function_calling"
)
expand_caption_model = qwen_llm_7b.with_structured_output(
    ExpandCaption, method="function_calling"
)

discriminator_model = qwen_llm_7b.with_structured_output(
    ResponseDiscriminator, method="function_calling"
)


def combine_query_terms(original_query: str, optimized_query: OptimizedQuery) -> str:
    """
    组合优化后的查询条件
    """
    combined_query = f"{original_query} {' '.join(optimized_query.expanded_terms)}"
    return combined_query.strip()


async def retriever_node(
    state: RAGState, config: RunnableConfig
) -> Command[Literal["expand_node"]]:
    rag_expert_name = Configuration.from_runnable_config(config).rag_expert_name

    # optimization_query
    query = state["messages"][-1].content
    try:
        optimization_query = await query_optimization_model.ainvoke(
            query_optimization_prompt.invoke(
                {"messages": [HumanMessage(content=query)]}
            )
        )
        combine_query = combine_query_terms(query, optimization_query)
    except Exception as e:
        combine_query = query

    # define compressor
    compressor = CrossEncoderReranker(model=bge_reranker, top_n=5)

    # small_chunk
    small_db = FAISS.load_local(
        vector_dir,
        bge_embeddings,
        index_name=f"{rag_expert_name}_small_chunk",
        allow_dangerous_deserialization=True,
    )
    small_retriever = small_db.as_retriever(search_kwargs={"k": 10})
    small_compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=small_retriever
    )
    small_docs = await small_compression_retriever.ainvoke(combine_query)

    # big_chunk
    big_db = FAISS.load_local(
        vector_dir,
        bge_embeddings,
        index_name=f"{rag_expert_name}_big_chunk",
        allow_dangerous_deserialization=True,
    )
    big_retriever = big_db.as_retriever(search_kwargs={"k": 10})
    big_compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=big_retriever
    )
    big_docs = await big_compression_retriever.ainvoke(combine_query)
    return Command(
        update={"query": query, "small_retrieve_text": small_docs, "big_retrieve_text": big_docs},
        goto="expand_node",
    )


async def expand_node(
    state: RAGState, config: RunnableConfig
) -> Command[Literal["generation_node"]]:
    mongo_client = await get_db()
    big_retrieve_text = state["big_retrieve_text"]
    history_caption = []
    db_expand_text = []
    for text in big_retrieve_text:
        source = text.metadata["source"]
        content = text.page_content
        try:
            expand_caption = await expand_caption_model.ainvoke(
                expand_caption_prompt.invoke(
                    {"messages": [HumanMessage(content=content)]}
                )
            )
            for caption in expand_caption.caption:
                if caption in history_caption:
                    continue
                expert_info = await mongo_client.find_one(
                    "pdf_info",
                    query={"source": source, "caption": {"$regex": caption}},
                    projection={"_id": 0},
                )
                if expert_info:
                    db_expand_text.append(expert_info)
                history_caption.append(caption)
        except Exception as e:
            continue
    return Command(
        update={"expand_text": db_expand_text},
        goto="generation_node",
    )


async def generation_node(
    state: RAGState, config: RunnableConfig
) -> Command[Literal["discriminator_node", "__end__"]]:
    small_retrieve_text = state["small_retrieve_text"]
    big_retrieve_text = state["big_retrieve_text"]
    expand_text = state["expand_text"]
    query = state["query"]
    rag_expert_name = Configuration.from_runnable_config(config).rag_expert_name
    mongo_client = await get_db()
    expert_info = await mongo_client.find_one(
        "rag_expert", query={"name": rag_expert_name}, projection={"_id": 0}
    )
    relevant_pdf = expert_info.get("rag_data", [])
    if relevant_pdf:
        relevant_pdf_images = []
        page_idx_history = []
        for doc in small_retrieve_text:
            source = doc.metadata.get("source")
            page_idx = doc.metadata.get("page_idx")
            if page_idx in page_idx_history:
                continue
            pdf_image_path = os.path.join(
                pdf_dir, os.path.splitext(source)[0], f"{str(page_idx).zfill(3)}.jpg"
            )
            relevant_pdf_images.append(pdf_image_path)
            page_idx_history.append(page_idx)

        relevant_big_chunk = []
        for doc in big_retrieve_text:
            relevant_big_chunk.append(doc.page_content)
        system_prompt = """
        【多模态问答生成指引】
            你是一个专业的多模态问答助手，请根据以下步骤处理用户请求：

            1. 信息整合阶段：
            <检索到的文档片段>
            '''
            {document_context}
            '''
            2. 交叉验证要求：
            • 识别文档与图像中相互印证的信息点
            • 标注图像中可视化但文本未提及的细节
            • 发现图文矛盾时标注并请求确认

            3. 回答生成规范：
            根据用户问题"{question}"，请：
            ① 先总结文本核心信息 (不超过3个要点)
            ② 再描述图像提供的视觉证据
            ③ 最后综合图文给出分步骤解答
            ④ 使用「根据文本资料」/「如图所示」等引用标注

            限制条件：
            ✘ 禁止臆测未验证的图文关系
            ✘ 避免直接引用超过100字的原文
            ✘ 当图像与文本冲突时主动澄清

            请确认理解上述要求，并生成最终回答。
        """
        info = []
        for big_chunk in relevant_big_chunk:
            info.append(f"{big_chunk}")
        for expand_info in expand_text:
            if expand_info.get("type") == "table":
                info.append(
                    f"# {expand_info.get('caption', '')}\n{expand_info.get('table_body', '')}"
                )
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    document_context="\n\n".join(info), question=query
                ),
            },
            {
                "role": "user",
                "content": [],
            },
        ]
        for image in relevant_pdf_images:
            messages[1]["content"].append({"type": "image_url", "image_url": image})
        history_img_path = []
        for expand_info in expand_text:
            img_path = expand_info.get("img_path", "")
            if img_path and img_path not in history_img_path:
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": img_path,
                    }
                )
                history_img_path.append(img_path)
        output = await qwen_25_vl.ainvoke(messages)
        output_text = output.content
        return_content = [{"type": "text", "text": output_text}]
        for input_contant_data in messages[1]["content"]:
            if input_contant_data["type"] == "image_url":
                return_content.append(input_contant_data)
        return Command(
            update={"response": return_content},
            goto="discriminator_node",
        )
    else:
        return Command(
            update={
                "messages": AIMessage(
                    content="无法基于当前的资料回答用户问题，请尝试增加资料或换用其他方式进行提问。"
                )
            },
            goto="__end__",
        )


async def discriminator_node(state: RAGState) -> Command[Literal["__end__"]]:
    system_prompt = """
        # 核心任务
        判断AI生成的回答是否有效解决用户提问，返回布尔值及简要理由

        # 有效性标准
        回答必须同时满足：
        1. 准确无误：无事实性错误或过时信息
        2. 切题相关：直接或间接回应用户提问的核心诉求
        3. 完整自洽：包含必要的关键信息且逻辑自洽

        # 输出格式
        使用ResponseDiscriminator格式化输出，包含以下字段：
        - is_valid：是否有效
        - reason：判断依据

        # 判断原则
        - 允许合理推断，但必须基于问题上下文
        - 部分正确视为无效
        - 存在矛盾陈述视为无效"

        用户输入：
        '''
        {input}
        '''

        模型回答：
        '''
        {response}
        '''
        """
    query = state["query"]
    return_content = state["response"]
    generate_response_text = [
        response["text"] for response in return_content if response["type"] == "text"
    ][0]
    model_input_message = [
            {
                "role": "system",
                "content": system_prompt.format(input=query, response=generate_response_text)
            }
        ]
    discrimination_result = await discriminator_model.ainvoke(model_input_message)
    if discrimination_result.is_valid:
        return Command(
            update={"messages": AIMessage(content=return_content)},
            goto="__end__",
        )
    else:
        return Command(
            update={
                "messages": AIMessage(
                    content="无法基于当前的资料回答用户问题，请尝试增加资料或换用其他方式进行提问。"
                )
            },
            goto="__end__",
        )


workflow = StateGraph(RAGState)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("expand_node", expand_node)
workflow.add_node("generation_node", generation_node)
workflow.add_node("discriminator_node", discriminator_node)
workflow.add_edge(START, "retriever_node")
rag_workflow = workflow.compile()
