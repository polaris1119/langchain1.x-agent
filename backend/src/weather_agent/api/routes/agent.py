"""
FastAPI Agent 路由
处理天气查询 Agent 的 API 请求
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from weather_agent.models.schemas import ChatRequest, ChatResponse
from weather_agent.agent.weather_agent import WeatherAgent
import uuid
import json


router = APIRouter(prefix="/chat", tags=["agent"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与天气查询 Agent 进行对话

    Args:
        request: 聊天请求，包含用户消息

    Returns:
        ChatResponse: Agent 的响应
    """
    try:
        # 创建 Agent 实例
        agent = WeatherAgent()

        # 调用 Agent 获取回复
        response = await agent.chat(request.message)

        # 返回响应
        return ChatResponse(
            response=response,
            session_id=request.session_id or str(uuid.uuid4())
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理请求时发生错误: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "Weather Agent API"}


async def _stream_event_generator(message: str, session_id: str):
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
    yield f"event: session_id\ndata: {json.dumps({'session_id': session_id})}\n\n"

    # 流式输出 AI 回复
    async for chunk in agent.astream_chat(message):
        if chunk:
            # SSE 格式: event: message\ndata: {"content": "..."}\n\n
            yield f"event: message\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

    # 发送结束事件
    yield "event: done\ndata: {}\n\n"


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
        session_id = request.session_id or str(uuid.uuid4())
        return StreamingResponse(
            _stream_event_generator(request.message, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理流式请求时发生错误: {str(e)}"
        )
