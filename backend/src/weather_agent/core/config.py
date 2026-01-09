"""
配置管理模块
负责加载和管理所有环境变量配置
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""

    # OpenRouter 配置
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "nex-agi/deepseek-v3.1-nex-n1:free"

    # 和风天气配置
    QWEATHER_API_KEY: str
    QWEATHER_BASE_URL: str = "https://devapi.qweather.com/v7"

    # FastAPI 配置
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Weather Agent API"

    # CORS 配置
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 日志配置
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# 全局配置实例
settings = Settings()
