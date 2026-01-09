# 🌤️ LangChain 天气查询 Agent

基于 LangChain 1.2.6 实现的智能天气查询 Agent，支持终端和 Web 两种测试界面。

## 项目特性

- 🤖 **智能对话**: 使用 LangChain Agent 实现自然语言天气查询
- 🌍 **城市支持**: 支持查询中国所有城市天气
- 📱 **双界面**: 提供终端 CLI 和 Web UI 两种交互方式
- ⚡ **最新技术**: 使用 LangChain 1.2.6、React 19、Vite 6 等最新技术栈
- 🎨 **现代 UI**: 基于 Tailwind CSS 和自定义组件的精美界面

## 技术栈

### 后端
- **Python 3.11+**
- **LangChain 1.2.6** - LLM 应用框架
- **OpenRouter** - LLM 服务提供商 (nex-agi/deepseek-v3.1-nex-n1:free)
- **FastAPI** - 现代化 Web 框架
- **和风天气 API** - 天气数据来源
- **uv** - Python 包管理器

### 前端
- **React 19** - UI 框架
- **Vite 6** - 构建工具
- **TypeScript 5** - 类型安全
- **Tailwind CSS 4** - 样式框架
- **Zustand 5** - 状态管理
- **Axios** - HTTP 客户端

## 项目结构

```
code/
├── backend/                      # Python 后端
│   ├── pyproject.toml            # uv 项目配置
│   ├── .env                      # 环境变量
│   └── src/
│       └── weather_agent/        # 主包 (src layout)
│           ├── __init__.py
│           ├── main.py           # FastAPI 入口
│           ├── cli.py            # 终端测试界面
│           ├── core/
│           │   └── config.py     # 配置管理
│           ├── agent/
│           │   └── weather_agent.py  # LangChain Agent
│           ├── services/
│           │   └── qweather_service.py  # 和风天气 API
│           ├── api/
│           │   └── routes/
│           │       └── agent.py  # REST API 路由
│           └── models/
│               └── schemas.py    # Pydantic 模型
│
└── frontend/                     # React 前端
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api/                  # API 客户端
        ├── components/           # UI 组件
        ├── store/                # Zustand 状态
        ├── types/                # TypeScript 类型
        └── pages/                # 页面组件
```

## 快速开始

### 1. 环境准备

确保已安装以下工具：

- **Python 3.11+**
- **uv** (Python 包管理器): `pip install uv`
- **pnpm** (Node.js 包管理器): `npm install -g pnpm`

### 2. 获取 API 密钥

- **OpenRouter API Key**: 访问 [OpenRouter](https://openrouter.ai/) 注册并获取
- **和风天气 API Key**: 访问 [和风天气开发平台](https://dev.qweather.com/) 注册并获取

### 3. 后端设置

```bash
# 进入后端目录
cd backend

# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥

# 运行 FastAPI 服务器
uv run uvicorn weather_agent.main:app --reload --port 8000
```

### 4. 前端设置

```bash
# 进入前端目录
cd frontend

# 安装依赖
pnpm install

# 配置环境变量
cp .env.example .env

# 运行开发服务器
pnpm dev
```

### 5. 访问应用

- **Web 界面**: http://localhost:5173
- **API 文档**: http://localhost:8000/docs

## 使用方法

### Web 界面

1. 启动后端和前端服务
2. 访问 http://localhost:5173
3. 输入天气查询，例如：
   - "北京今天天气怎么样？"
   - "上海未来三天天气"
   - "深圳现在多少度？"

### 终端界面

```bash
cd backend
uv run python -m weather_agent.cli
```

然后输入你的问题即可。

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

## 环境变量说明

### 后端 (.env)

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | - |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址 | https://openrouter.ai/api/v1 |
| `OPENROUTER_MODEL` | 使用的模型 | nex-agi/deepseek-v3.1-nex-n1:free |
| `QWEATHER_API_KEY` | 和风天气 API 密钥 | - |
| `QWEATHER_BASE_URL` | 和风天气 API 地址 | https://devapi.qweather.com/v7 |

### 前端 (.env)

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VITE_API_URL` | 后端 API 地址 | /api |

## 开发命令

### 后端

```bash
# 安装依赖
uv sync

# 运行开发服务器
uv run uvicorn weather_agent.main:app --reload --port 8000

# 运行终端测试
uv run python -m weather_agent.cli
```

### 前端

```bash
# 安装依赖
pnpm install

# 运行开发服务器
pnpm dev

# 构建生产版本
pnpm build

# 预览生产版本
pnpm preview
```

## 技术亮点

- **KISS**: 简洁的代码结构，易于理解和维护
- **DRY**: 复用组件和工具函数，避免重复
- **SOLID**: 清晰的模块职责分离
- **类型安全**: 全栈类型定义（Python type hints + TypeScript）

## 许可证

MIT

## 致谢

- [LangChain](https://www.langchain.com/) - 强大的 LLM 应用框架
- [OpenRouter](https://openrouter.ai/) - 统一的 LLM API 接口
- [和风天气](https://www.qweather.com/) - 专业的天气数据服务
