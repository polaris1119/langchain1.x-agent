# 天气查询助手 - 前端

基于 React 19 + Vite 6 + TypeScript 5 的现代化天气查询 Web 应用。

## 项目特性

- ⚡ **Vite 6** - 极速的开发体验
- ⚛️ **React 19** - 最新的 React 特性
- 🎨 **Tailwind CSS 4** - 现代化的样式框架
- 📦 **Zustand** - 轻量级状态管理
- 🔒 **TypeScript** - 完整的类型安全
- 🛣️ **React Router 7** - 路由管理

## 项目结构

```
frontend/
├── src/
│   ├── main.tsx              # 应用入口
│   ├── App.tsx               # 根组件（路由配置）
│   ├── api/
│   │   ├── client.ts         # Axios 客户端配置
│   │   └── agent.ts          # Agent API 接口
│   ├── components/
│   │   ├── ui/               # 基础 UI 组件
│   │   └── WeatherChat.tsx   # 天气聊天组件
│   ├── store/
│   │   └── weatherStore.ts   # Zustand 状态管理
│   ├── types/
│   │   └── agent.ts          # TypeScript 类型定义
│   ├── lib/
│   │   └── utils.ts          # 工具函数
│   └── pages/
│       └── WeatherPage.tsx   # 天气页面
├── index.html
├── vite.config.ts            # Vite 配置
├── tailwind.config.js        # Tailwind CSS 配置
├── tsconfig.json             # TypeScript 配置
└── package.json
```

## 快速开始

### 1. 安装依赖

```bash
pnpm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# 后端 API 地址（开发时使用代理，默认 /api 即可）
VITE_API_URL=/api
```

### 3. 启动开发服务器

```bash
pnpm dev
```

访问 http://localhost:5173

### 4. 构建生产版本

```bash
pnpm build
```

### 5. 预览生产版本

```bash
pnpm preview
```

## 核心组件

### WeatherChat

主要的聊天交互组件，功能包括：

- 💬 实时消息显示
- ⌨️ 输入框和发送按钮
- ⏳ 加载状态指示
- ⚠️ 错误提示
- 📜 自动滚动到最新消息
- 🎯 支持回车键发送

### weatherStore (Zustand)

状态管理，包含：

```typescript
interface WeatherState {
  messages: ChatMessage[]    // 消息列表
  isLoading: boolean         // 加载状态
  error: string | null       // 错误信息

  addMessage(message)        // 添加消息
  setLoading(loading)        // 设置加载状态
  setError(error)            // 设置错误
  clearMessages()            // 清空消息
}
```

### agentApi

API 调用封装：

```typescript
// 发送聊天消息
await agentApi.chat(message)

// 健康检查
await agentApi.healthCheck()
```

## 样式系统

### Tailwind CSS

项目使用 Tailwind CSS 进行样式开发：

- **配置文件**: `tailwind.config.js`
- **基础样式**: `src/index.css`
- **工具函数**: `cn()` 用于合并类名

### 自定义组件

基础 UI 组件位于 `src/components/ui/`：

- `Button` - 按钮组件
- `Card` - 卡片容器
- `Input` - 输入框
- `ScrollArea` - 滚动区域

## 开发指南

### 添加新页面

1. 在 `src/pages/` 创建页面组件
2. 在 `src/App.tsx` 添加路由

```tsx
<Route path="/new-page" element={<NewPage />} />
```

### 添加新 API

1. 在 `src/types/` 定义类型
2. 在 `src/api/` 添加 API 函数

```typescript
// src/api/example.ts
import apiClient from './client'

export const exampleApi = {
  async getData() {
    const { data } = await apiClient.get('/example')
    return data
  }
}
```

### 状态管理

使用 Zustand 创建新的 store：

```typescript
import { create } from 'zustand'

interface MyState {
  data: string
  setData: (data: string) => void
}

export const useMyStore = create<MyState>((set) => ({
  data: '',
  setData: (data) => set({ data }),
}))
```

## 路径别名

项目配置了 `@` 别名指向 `src` 目录：

```tsx
// 推荐
import { Button } from '@/components/ui/button'

// 不推荐
import { Button } from '../../../components/ui/button'
```

## Vite 配置

### 代理配置

开发环境下，Vite 会代理 API 请求到后端：

```typescript
// vite.config.ts
server: {
  port: 5173,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 代码规范

### TypeScript

- 启用严格模式
- 所有组件和函数都有类型定义
- 使用接口定义数据结构

### 命名规范

- 组件：PascalCase（如 `WeatherChat`）
- 函数：camelCase（如 `addMessage`）
- 类型：PascalCase（如 `ChatMessage`）
- 常量：UPPER_SNAKE_CASE（如 `API_URL`）

### 文件组织

- 每个组件一个文件
- 相关的文件放在同一目录
- 使用 `index.ts` 导出模块

## 常用命令

```bash
# 安装依赖
pnpm install

# 添加依赖
pnpm add <package-name>
pnpm add -D <dev-package-name>

# 开发
pnpm dev

# 构建
pnpm build

# 预览
pnpm preview

# 代码检查
pnpm lint
```

## 浏览器支持

- Chrome (最新版)
- Firefox (最新版)
- Safari (最新版)
- Edge (最新版)

## 许可证

MIT
