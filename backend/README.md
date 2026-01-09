# Weather Agent Backend

基于 LangChain 1.2.6 的智能天气查询 Agent 后端服务。

## 项目结构

采用 [src layout](https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#using-src-layout) 规范：

```
backend/
├── pyproject.toml           # uv 项目配置
├── .env                     # 环境变量
└── src/
    └── weather_agent/       # 主包
        ├── __init__.py
        ├── main.py          # FastAPI 应用入口
        ├── cli.py           # 终端测试界面
        ├── core/
        │   └── config.py    # 配置管理
        ├── agent/
        │   └── weather_agent.py  # LangChain Agent
        ├── services/
        │   └── qweather_service.py  # 和风天气 API
        ├── api/
        │   └── routes/
        │       └── agent.py  # REST API 路由
        └── models/
            └── schemas.py   # Pydantic 数据模型
```

## 技术栈

- **Python 3.11+**
- **LangChain 1.2.3+** - LLM 应用框架
- **FastAPI** - 现代化 Web 框架
- **Pydantic 2+** - 数据验证
- **httpx** - 异步 HTTP 客户端
- **uv** - Python 包管理器

## 快速开始

### 1. 安装 uv

```bash
pip install uv
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入以下必要的环境变量：

```bash
# OpenRouter 配置（必需）
OPENROUTER_API_KEY=your_openrouter_api_key_here

# 和风天气配置（必需）
QWEATHER_API_KEY=your_qweather_api_key_here
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 运行服务

#### FastAPI 服务器

```bash
uv run uvicorn weather_agent.main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

#### 终端测试界面

```bash
uv run python -m weather_agent.cli
```

## API 端点

### 聊天接口

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "北京今天天气怎么样？"
}
```

**响应**:
```json
{
  "response": "📍 北京当前天气：\n🌡️ 温度: 15°C...",
  "session_id": "uuid"
}
```

### 健康检查

```http
GET /api/v1/chat/health
```

## 开发

### 运行开发服务器

```bash
# 热重载模式
uv run uvicorn weather_agent.main:app --reload --port 8000

# 指定日志级别
uv run uvicorn weather_agent.main:app --reload --log-level debug
```

### 添加依赖

```bash
# 添加生产依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>
```

## 环境变量

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | - | ✅ |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址 | https://openrouter.ai/api/v1 | ❌ |
| `OPENROUTER_MODEL` | 使用的模型 | nex-agi/deepseek-v3.1-nex-n1:free | ❌ |
| `QWEATHER_API_KEY` | 和风天气 API 密钥 | - | ✅ |
| `QWEATHER_BASE_URL` | 和风天气 API 地址 | https://devapi.qweather.com/v7 | ❌ |
| `API_V1_PREFIX` | API v1 前缀 | /api/v1 | ❌ |
| `CORS_ORIGINS` | CORS 允许的来源 | ["http://localhost:5173"] | ❌ |

## 架构说明

### LangChain Agent

使用 LangChain 的 Tool-Calling Agent 架构：

1. **LLM**: 通过 OpenRouter 调用 DeepSeek 模型
2. **Tools**: `get_weather` 工具用于查询天气
3. **Executor**: AgentExecutor 负责执行推理和工具调用

### 服务层

- **QWeatherService**: 封装和风天气 API 调用
  - `get_city_id()`: 城市名称 → 城市ID
  - `get_current_weather()`: 获取实时天气
  - `get_forecast()`: 获取天气预报
  - `format_weather_text()`: 格式化为自然语言

### API 层

- **FastAPI**: 提供 REST API
- **CORS**: 支持前端跨域请求
- **自动文档**: Swagger UI 和 ReDoc

## 许可证

MIT
