"""
LangChain 天气查询 Agent
使用 LangChain 框架实现智能天气查询助手
"""
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import AIMessage, HumanMessage
from weather_agent.core.config import settings
from weather_agent.services.qweather_service import QWeatherService
from typing import AsyncIterator


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

    def _create_inputs(self, message: str) -> dict:
        """
        创建 Agent 输入格式（使用 LangChain 1.x Message 类型）

        Args:
            message: 用户消息

        Returns:
            Agent 输入字典
        """
        return {
            "messages": [HumanMessage(content=message)]
        }

    def _extract_content_from_message(self, message) -> str | None:
        """
        从 Message 对象中提取内容（符合 LangChain 1.x 最佳实践）

        Args:
            message: LangChain Message 对象（AIMessage、HumanMessage 等）

        Returns:
            提取的内容字符串，如果无法提取则返回 None
        """
        # 优先使用 content 属性（LangChain 1.x 标准）
        if hasattr(message, "content"):
            content = message.content
            # 处理不同类型的 content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and content:
                # 处理 content_blocks 格式
                if hasattr(content[0], "text"):
                    return "".join(block.text for block in content if hasattr(block, "text"))
                # 处理字符串列表
                elif all(isinstance(item, str) for item in content):
                    return "".join(content)
        return None

    async def astream_chat(self, message: str) -> AsyncIterator[str]:
        """
        与 Agent 进行流式对话（核心方法，符合 LangChain 1.x 最佳实践）

        Args:
            message: 用户消息

        Yields:
            str: Agent 的回复片段
        """
        try:
            inputs = self._create_inputs(message)

            # 使用 astream 的 updates 模式获取状态更新
            async for chunk in self.agent.astream(inputs, stream_mode="updates"):
                # chunk 格式: {node_name: {"messages": [Message]}}
                for node_data in chunk.values():
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        for msg in messages:
                            # 仅输出 AI 的回复消息
                            if isinstance(msg, AIMessage):
                                content = self._extract_content_from_message(msg)
                                if content:
                                    yield content

        except Exception as e:
            yield f"\n\n[错误] 处理请求时出现错误：{str(e)}"

    async def chat(self, message: str) -> str:
        """
        与 Agent 进行对话（非流式，内部使用流式逻辑）

        Args:
            message: 用户消息

        Returns:
            Agent 的完整回复
        """
        try:
            # 收集所有流式片段
            full_response = ""
            async for chunk in self.astream_chat(message):
                full_response += chunk

            return full_response if full_response else "抱歉，未能生成回复。"

        except Exception as e:
            return f"抱歉，处理请求时出现错误：{str(e)}"
