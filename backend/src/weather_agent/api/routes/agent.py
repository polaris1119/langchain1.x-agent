"""
FastAPI Agent 路由
处理天气查询 Agent 的 API 请求
"""
from fastapi import APIRouter, HTTPException, status
from weather_agent.models.schemas import ChatRequest, ChatResponse
from weather_agent.agent.weather_agent import WeatherAgent
import uuid


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
