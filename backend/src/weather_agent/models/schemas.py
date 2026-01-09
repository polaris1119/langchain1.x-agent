"""
Pydantic 数据模型
定义 API 请求和响应的数据结构
"""
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    session_id: str
