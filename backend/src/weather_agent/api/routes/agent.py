"""
FastAPI Agent 路由
处理天气查询 Agent 的 API 请求

核心业务逻辑在 WeatherAgent 中实现，路由层仅负责：
- HTTP 请求/响应转换
- SSE 流式输出编码
- 错误处理和状态码转换
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from weather_agent.models.schemas import ChatRequest, ChatResponse
from weather_agent.agent.weather_agent import WeatherAgent
import uuid
import json
from typing import AsyncGenerator


router = APIRouter(prefix="/chat", tags=["agent"])


def _get_or_create_session_id(session_id: str | None) -> str:
    """获取或创建会话 ID"""
    return session_id or str(uuid.uuid4())


def _sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与天气查询 Agent 进行对话（非流式）

    内部使用流式逻辑收集完整响应后一次性返回，
    确保与流式接口的行为完全一致。

    Args:
        request: 聊天请求，包含用户消息

    Returns:
        ChatResponse: Agent 的完整响应
    """
    try:
        agent = WeatherAgent()
        response = await agent.chat(request.message)
        session_id = _get_or_create_session_id(request.session_id)

        return ChatResponse(response=response, session_id=session_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理请求时发生错误: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "Weather Agent API"}


async def _stream_event_generator(
    message: str,
    session_id: str
) -> AsyncGenerator[str, None]:
    """
    SSE 事件生成器

    Args:
        message: 用户消息
        session_id: 会话 ID

    Yields:
        SSE 格式的数据流
    """
    agent = WeatherAgent()

    # 发送 session_id 作为第一个事件
    yield _sse_event("session_id", {"session_id": session_id})

    # 流式输出 AI 回复
    async for chunk in agent.astream_chat(message):
        if chunk:
            yield _sse_event("message", {"content": chunk})

    # 发送结束事件
    yield _sse_event("done", {})


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    与天气查询 Agent 进行流式对话 (SSE)

    Args:
        request: 聊天请求，包含用户消息

    Returns:
        StreamingResponse: SSE 流式响应
    """
    try:
        session_id = _get_or_create_session_id(request.session_id)
        return StreamingResponse(
            _stream_event_generator(request.message, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理流式请求时发生错误: {str(e)}"
        )
