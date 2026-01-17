# LangChain 1.x Agent 示例

> 基于 LangChain 1.x 的 AI Agent 示例项目，演示如何使用 OpenRouter API（通过 GLM-4.5-Air 模型）创建工具调用 Agent。

## 项目背景

LangChain 1.x 是一次重大架构升级，从"堆 API 的工具箱"演变为"以 Agent 为中心的工程化框架"。这个项目展示了 LangChain 1.x 的六大核心特性：

### 1. Agent 统一入口
```python
from langchain.agents import create_agent
agent = create_agent(model=llm, tools=tools, system_prompt="...")
```

### 2. Middleware 成为一等公民
```python
from langchain.agents.middleware import before_model, after_model

@before_model
def my_middleware(state: AgentState, runtime: Runtime) -> dict | None:
    # 在模型调用前执行
    ...
```

### 3. 默认强绑定 LangGraph
- 有状态、有回放、有流式
- 返回 `CompiledStateGraph`，无需手动"画图"

### 4. 统一内容表示
- 使用 Content Blocks（内容块）统一处理各类消息

### 5. 结构化输出
- 一次调用就返回类型安全对象

### 6. 命名空间变干净
- 新旧 API 彻底分家
- 老的迁移到 `langchain-classic`，本项目不使用

## 环境要求

| 项目 | 版本要求 |
|------|----------|
| Python | 3.11+ |
| uv | 最新版（推荐用于依赖管理） |

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> **提示**：获取 OpenRouter API Key 请访问 [https://openrouter.ai/](https://openrouter.ai/)

### 3. 运行示例

```bash
# 基础 Agent 示例
uv run agent01.py

# 带 Middleware 的 Agent 示例
uv run agent02.py
```

## 示例说明

### agent01.py - 基础 Agent

演示最基本的 Agent 创建和工具调用流程：

- 使用 `@tool` 装饰器定义工具
- 使用 `create_agent()` 创建 Agent
- 调用 Agent 并获取结果
- 打印完整的工具调用轨迹

**核心代码片段：**
```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_current_weather(city: str) -> str:
    """根据城市名返回当前天气信息"""
    ...

agent = create_agent(
    model=llm,
    tools=[get_current_weather],
    system_prompt="你是一个乐于助人的中文 AI 助手。"
)

result = agent.invoke({"messages": [("user", "帮我查一下北京的天气")]})
```

### agent02.py - Agent + Middleware

演示如何使用中间件增强 Agent 功能：

- 使用 `@before_model` 和 `@after_model` 装饰器定义中间件
- 中间件自动在模型调用前后执行
- 实现日志记录、监控等横切关注点

**核心代码片段：**
```python
from langchain.agents.middleware import before_model, after_model

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    """在模型调用前执行"""
    print(f"[LOG] 调用次数: {_call_count}")
    print(f"[LOG] 消息总数: {len(state['messages'])}")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> dict | None:
    """在模型调用后执行"""
    if hasattr(last, "tool_calls") and last.tool_calls:
        print(f"[LOG] 模型决定调用工具: {...}")
    return None

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="...",
    middleware=[log_before_model, log_after_model]  # 添加中间件
)
```

## 项目结构

```
langchain1.x-agent/
├── .env                  # 环境变量配置（不提交到版本控制）
├── .gitignore           # Git 忽略规则
├── .python-version      # Python 版本（uv 使用）
├── CLAUDE.md            # Claude Code 项目指导文件
├── LICENSE              # 开源协议
├── pyproject.toml       # 项目配置和依赖声明
├── README.md            # 项目说明文档
├── agent01.py           # 基础 Agent 示例
└── agent02.py           # Agent + Middleware 示例
```

## 技术栈

| 依赖 | 版本 | 用途 |
|------|------|------|
| langchain | 1.2.3+ | 核心 Agent 框架 |
| langchain-openai | 1.1.7+ | OpenRouter 兼容适配 |
| python-dotenv | 0.9.9+ | 环境变量管理 |

## 开发建议

### 依赖管理
推荐使用 `uv` 替代传统的 pip/venv：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 运行脚本
uv run <script.py>
```

### 代码风格
- 使用 `from __future__ import annotations` 启用延迟类型注解
- 遵循 PEP 8 编码规范
- 工具函数使用 `@tool` 装饰器，并编写清晰的 docstring

## 常见问题

**Q: 为什么使用 OpenRouter 而不是直接调用 OpenAI？**

A: OpenRouter 提供统一的 API 接口访问多种 LLM，本项目使用免费的 GLM-4.5-Air 模型进行演示。

**Q: 中间件可以修改状态吗？**

A: 可以。`@before_model` 和 `@after_model` 可以返回一个字典来更新状态，返回 `None` 则不修改。

**Q: 如何使用其他 LLM？**

A: 修改 `ChatOpenAI` 的 `model` 和 `base_url` 参数即可，确保 LLM 支持工具调用。

## 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenRouter 官网](https://openrouter.ai/)

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
