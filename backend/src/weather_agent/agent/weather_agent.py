"""
LangChain 天气查询 Agent
使用 LangChain 框架实现智能天气查询助手
"""
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from weather_agent.core.config import settings
from weather_agent.services.qweather_service import QWeatherService


# 创建天气查询工具
@tool
async def get_weather(location: str, days: int = 1) -> str:
    """
    查询指定城市的天气信息

    Args:
        location: 城市名称，例如"北京"、"上海"、"广州"
        days: 查询天数，1=今天，2=今天和明天，最多7天

    Returns:
        天气信息的自然语言描述
    """
    service = QWeatherService()
    try:
        result = await service.format_weather_text(location, days)
        return result
    finally:
        await service.close()


class WeatherAgent:
    """天气查询 Agent 类"""

    def __init__(self):
        """初始化 Agent"""
        # 初始化 LLM（使用 OpenRouter）
        self.llm = ChatOpenAI(
            model=settings.OPENROUTER_MODEL,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            temperature=0.7,
        )

        # 定义工具列表
        self.tools = [get_weather]

        # 创建 Agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt="你是一个专业的天气助手，可以帮助用户查询中国任何城市的天气信息。"
                         "使用 get_weather 工具来获取天气数据，然后用自然、友好的方式回复用户。"
                         "如果用户没有明确指定天数，默认查询当天天气。",
            debug=False,
        )

    async def chat(self, message: str) -> str:
        """
        与 Agent 进行对话

        Args:
            message: 用户消息

        Returns:
            Agent 的回复
        """
        try:
            inputs = {
                "messages": [
                    {"role": "user", "content": message}
                ]
            }

            # 调用 agent
            response = await self.agent.ainvoke(inputs)

            # 提取最后一条 AI 消息作为回复
            messages = response.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, dict):
                    if msg.get("role") == "assistant":
                        return msg.get("content", "")
                elif hasattr(msg, "content"):
                    return str(msg.content)

            return "抱歉，未能生成回复。"

        except Exception as e:
            return f"抱歉，处理请求时出现错误：{str(e)}"

    async def astream_chat(self, message: str):
        """
        与 Agent 进行流式对话

        Args:
            message: 用户消息

        Yields:
            str: Agent 的回复片段
        """
        try:
            inputs = {
                "messages": [
                    {"role": "user", "content": message}
                ]
            }

            # 使用 astream 进行流式输出
            async for chunk in self.agent.astream(inputs, stream_mode="updates"):
                # chunk 格式: {'agent': {'messages': [AIMessage(content='...')]}}
                if isinstance(chunk, dict):
                    for value in chunk.values():
                        if isinstance(value, dict) and "messages" in value:
                            messages = value["messages"]
                            for msg in messages:
                                if hasattr(msg, "content") and msg.content:
                                    yield str(msg.content)
                        elif isinstance(value, str) and value:
                            yield value
                elif isinstance(chunk, str) and chunk:
                    yield chunk

        except Exception as e:
            yield f"\n\n[错误] 处理请求时出现错误：{str(e)}"
