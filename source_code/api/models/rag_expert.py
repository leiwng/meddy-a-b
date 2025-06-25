from langchain_core.callbacks import file
from pydantic import BaseModel
from typing import Optional, List

# 用于PATCH请求的部分更新模型
class RagExpertUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class RAGDataResponse(BaseModel):
    file_path: str
    file_id: str
    task_id: str
    create_time: str

class RagExpertResponse(BaseModel):
    name: str
    description: str
    uuid: str
    rag_data: List[RAGDataResponse]
