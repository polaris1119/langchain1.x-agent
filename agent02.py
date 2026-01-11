from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
    after_model,
)
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

# 1. 加载环境变量（包含 OPENROUTER_API_KEY）
load_dotenv()

# 从环境变量获取 OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# 2. 定义工具函数

@tool
def get_current_weather(city: str) -> str:
    """获取指定城市的当前天气。"""
    city = city.lower()
    if "beijing" in city or "北京" in city:
        return "北京当前天气：晴，-4℃，空气质量良。"
    if "shanghai" in city or "上海" in city:
        return "上海当前天气：多云，2℃，有阵风。"
    return f"未找到 {city} 的天气信息。"


@tool
def add_numbers(a: float, b: float) -> str:
    """计算两个数的和。"""
    return str(a + b)


# 3. 自定义中间件（使用 LangChain 1.x 装饰器风格）

# 用于跟踪调用次数
_call_count = 0


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    在模型调用前执行的钩子。

    打印当前消息列表的统计信息和最后一条消息的预览。
    """
    global _call_count
    _call_count += 1

    messages = state.get("messages", [])

    print("=== Middleware: before_model ===")
    print(f"[LOG] 调用次数: {_call_count}")
    print(f"[LOG] 消息总数: {len(messages)}")

    if not messages:
        print("[LOG] 当前没有消息。")
        return None

    last = messages[-1]

    # 获取消息类型和内容
    msg_type = getattr(last, "type", "unknown")
    content = getattr(last, "content", "")

    # 截断过长的内容
    content_preview = (
        str(content)[:80] + "..." if len(str(content)) > 80 else str(content)
    )
    print(f"[LOG] 最近一条消息（type={msg_type}）: {content_preview}")

    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    在模型调用后执行的钩子。

    打印模型响应信息，包括是否调用了工具。
    """
    print("=== Middleware: after_model ===")

    messages = state.get("messages", [])
    if not messages:
        print("[LOG] 没有消息记录。")
        return None

    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        print(f"[LOG] 模型决定调用工具: {[tc.get('name') for tc in last.tool_calls]}")
    else:
        print("[LOG] 模型直接返回回答（未调用工具）")

    return None


def build_agent():
    """
    构建带有工具和中间件的 Agent。

    使用 LangChain 1.x 的 create_agent 函数：
    - system_prompt: 静态系统提示词（直接传字符串）
    - middleware: 中间件列表（用于日志、监控等）
    """
    # 初始化聊天模型
    llm = ChatOpenAI(
        model="z-ai/glm-4.5-air:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
    )

    # 准备工具
    tools = [get_current_weather, add_numbers]

    # 创建 Agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个乐于助人的中文 AI 助手。"
            "当用户询问天气或需要计算时，请优先调用相应的工具，然后再用自然语言总结结果。"
        ),
        middleware=[log_before_model, log_after_model],
    )

    return agent


def print_execution_details(messages: list[BaseMessage]) -> None:
    """打印执行过程的详细信息。"""
    print("\n" + "=" * 50)
    print("【中间步骤（工具调用轨迹）】")
    print("=" * 50)

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
            result_preview = (
                msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            )
            print(f"\n  工具返回: {result_preview}")

    if step_count == 0:
        print("  (本次调用未使用工具)")


def main():
    """
    主函数：演示 Agent + Middleware 的使用。

    中间件会自动在 Agent 执行过程中被调用，无需手动触发。
    """
    print("=" * 50)
    print("LangChain 1.x Agent + Middleware 示例")
    print("=" * 50)

    # 构建并运行 Agent（中间件已集成）
    agent = build_agent()

    # 测试用例
    test_inputs = [
        "帮我查一下北京的天气，然后再算一下 12.5 加 7.3。",
        "上海的天气怎么样？另外帮我算 100 减 37.5。",
    ]

    for user_input in test_inputs:
        print("\n" + "=" * 50)
        print(f"【用户输入】{user_input}")
        print("=" * 50)

        # 调用 Agent（中间件会自动执行）
        result = agent.invoke({"messages": [("user", user_input)]})

        # 打印最终回答
        messages = result.get("messages", [])
        if messages:
            print("\n【Agent 最终回答】")
            last_message = messages[-1]
            print(last_message.content)

            # 打印执行详情
            print_execution_details(messages)


if __name__ == "__main__":
    main()
