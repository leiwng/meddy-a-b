from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
import os

# 导入langchain相关库
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import sys

sys.path.append("/data/cjl/project/my_agent")

from source_code.api.config import vector_dir

OLLAMA_URL = os.environ.get("OLLAMA_URL", "")

bge_embeddings = OllamaEmbeddings(
    model="quentinz/bge-large-zh-v1.5:latest", base_url=OLLAMA_URL
)


class FAISSManager:
    """
    FAISS向量数据库管理类，用于文档的向量化存储和检索
    """

    def __init__(
        self,
        embedding_model: Optional[OllamaEmbeddings] = bge_embeddings,
        default_persist_dir: Optional[Path | str] = str(vector_dir),
        index_name: Optional[str] = "default_index",
    ):
        """
        初始化FAISS管理器

        Args:
            embedding_model: 嵌入模型，如果为None则使用默认的HuggingFaceEmbeddings
            default_persist_dir: 默认的向量库持久化目录，如果为None则使用'vectorstore'
            index_name: 向量库名称，如果为None则使用'default_index'
        """
        # 设置嵌入模型
        self.embedding_model = embedding_model

        # 设置默认持久化目录
        self.default_persist_dir = Path(default_persist_dir)

        # 设置向量库名称
        self.index_name = index_name

        # 初始化向量存储
        self.vectorstore = None

    def create_from_texts(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> FAISS:
        """
        从文本列表创建FAISS向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表，与texts一一对应

        Returns:
            FAISS: 创建的FAISS向量存储实例
        """
        self.vectorstore = FAISS.from_texts(
            texts=texts, embedding=self.embedding_model, metadatas=metadatas
        )
        return self.vectorstore

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        从Document对象列表创建FAISS向量存储

        Args:
            documents: Document对象列表

        Returns:
            FAISS: 创建的FAISS向量存储实例
        """
        self.vectorstore = FAISS.from_documents(
            documents=documents, embedding=self.embedding_model
        )
        return self.vectorstore

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        向现有的向量存储中添加文本

        Args:
            texts: 要添加的文本列表
            metadatas: 元数据列表，与texts一一对应

        Returns:
            List[str]: 添加的文档ID列表

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )

        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def delete_texts(self, ids: List[str]) -> None:
        """删除向量库中的文本

        Args:
            ids (List[str]): 要删除的文档ID列表

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )
        self.vectorstore.delete(ids=ids)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        执行相似度搜索

        Args:
            query: 查询文本
            k: 返回的最相似文档数量

        Returns:
            List[Document]: 最相似的文档列表

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )

        return self.vectorstore.similarity_search(query=query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        执行相似度搜索并返回相似度分数

        Args:
            query: 查询文本
            k: 返回的最相似文档数量

        Returns:
            List[Tuple[Document, float]]: 文档和相似度分数的元组列表

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )

        return self.vectorstore.similarity_search_with_score(query=query, k=k)

    def save_local(self, folder_path: Optional[Union[str, Path]] = None) -> Path:
        """
        将向量存储保存到本地

        Args:
            folder_path: 保存的文件夹路径，如果为None则使用默认路径

        Returns:
            Path: 保存的文件夹路径

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )

        if folder_path is None:
            folder_path = self.default_persist_dir
        else:
            folder_path = Path(folder_path)

        # 确保目录存在
        folder_path.mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(
            folder_path=str(folder_path), index_name=self.index_name
        )
        return folder_path

    def load_local(self, folder_path: Optional[Union[str, Path]] = None) -> FAISS:
        """
        从本地加载向量存储

        Args:
            folder_path: 加载的文件夹路径，如果为None则使用默认路径

        Returns:
            FAISS: 加载的FAISS向量存储实例

        Raises:
            FileNotFoundError: 当指定的文件夹不存在时
        """
        if folder_path is None:
            folder_path = self.default_persist_dir
        else:
            folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"找不到向量存储文件夹: {folder_path}")

        self.vectorstore = FAISS.load_local(
            folder_path=str(folder_path),
            embeddings=self.embedding_model,
            index_name=self.index_name,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    def merge_from(self, other_faiss: FAISS) -> None:
        """
        合并另一个FAISS向量存储到当前存储

        Args:
            other_faiss: 要合并的FAISS向量存储

        Raises:
            ValueError: 当向量存储未初始化时
        """
        if self.vectorstore is None:
            raise ValueError(
                "向量存储未初始化，请先调用create_from_texts或create_from_documents方法"
            )

        self.vectorstore.merge_from(other_faiss)


if __name__ == "__main__":
    # 创建一个FAISS向量存储
    faiss_manager = FAISSManager(
        default_persist_dir="/data/cjl/deploy/langgraph-api-data/static/file/vector",
        index_name="测试新专家_big_chunk",
    )

    faiss_manager.load_local()
    result = faiss_manager.similarity_search(query="介绍？", k=10)
    for result in result:
        print(result)
        print("=" * 40)
