"""
FastAPI 应用入口
启动和配置 Web 服务器
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from weather_agent.core.config import settings
from weather_agent.api.routes import agent


# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="基于 LangChain 的智能天气查询 Agent API",
    version="0.1.0",
)


# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册路由
app.include_router(agent.router, prefix=settings.API_V1_PREFIX)


# 根路径
@app.get("/")
async def root():
    """API 根路径"""
    return {
        "message": "Weather Agent API",
        "version": "0.1.0",
        "docs": "/docs",
    }


# 健康检查
@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "weather_agent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
