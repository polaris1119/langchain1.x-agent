from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 1. 加载环境变量（包含 OPENROUTER_API_KEY）
load_dotenv()

# 从环境变量获取 OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# 2. 定义一个"天气查询"工具（这里只是模拟）
@tool
def get_current_weather(city: str) -> str:
    """根据城市名返回当前天气信息（示例工具，内部数据是写死的）"""
    city = city.lower()
    if "beijing" in city or "北京" in city:
        return "北京当前天气：晴，-4℃，空气质量良。"
    if "shanghai" in city or "上海" in city:
        return "上海当前天气：多云，2℃，有阵风。"
    return f"{city} 当前天气信息暂不可用，请稍后再试。"


def main():
    # 3. 初始化聊天模型
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
    )

    # 4. 准备工具
    tools = [get_current_weather]

    # 5. 使用 create_agent 创建 Agent
    # 返回 CompiledStateGraph（基于 LangGraph）
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个乐于助人的中文 AI 助手。"
            "当用户询问与天气相关的问题时，请优先考虑调用相应工具。"
        ),
    )

    # 6. 准备输入状态
    user_input = "帮我查一下北京的天气，然后再用一句话建议我要不要带伞。"

    # 7. 调用 Agent（使用 invoke 方法）
    # LangChain 1.x 的状态包含 messages 字段
    result = agent.invoke({"messages": [("user", user_input)]})

    # 8. 打印最终回答
    # result 是一个状态字典，包含 messages 字段
    messages = result.get("messages", [])
    if messages:
        print("【Agent 最终回答】")
        # 最后一条消息通常是 AI 的回复
        last_message = messages[-1]
        print(last_message.content)

        # 9. 打印中间步骤（工具调用轨迹）
        print("\n【中间步骤（工具调用轨迹）】")
        step_count = 0
        for msg in messages:
            # 检查是否是 AI 消息且包含工具调用
            if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    step_count += 1
                    print(f"\nStep {step_count}:")
                    print(f"  工具名: {tool_call.get('name', 'unknown')}")
                    print(f"  工具参数: {tool_call.get('args', {})}")
            # 检查是否是工具返回消息
            elif msg.type == "tool":
                print(f"  工具返回: {msg.content}")

        if step_count == 0:
            print("  (本次调用未使用工具)")

        print("\n【完整消息历史】")
        for msg in messages:
            content_preview = (
                f"  {msg.type}: {msg.content[:100]}..."
                if len(msg.content) > 100
                else f"  {msg.type}: {msg.content}"
            )
            print(content_preview)


if __name__ == "__main__":
    main()
