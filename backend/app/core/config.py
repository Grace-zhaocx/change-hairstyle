from typing import List, Optional, Any, Dict, Union
from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):
    # 基础配置
    PROJECT_NAME: str = "AI Hairstyle API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # 环境配置
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]

    # 数据库配置
    DATABASE_URL: str = "sqlite+aiosqlite:///./hairstyle.db"

    # Redis配置
    REDIS_URL: str = "redis://localhost:6379"

    # MinIO配置
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "hairstyle"
    MINIO_SECURE: bool = False

    # JWT配置
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24小时

    # 文件上传配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    UPLOAD_DIR: str = "uploads"

    # AI模型配置
    MODEL_CACHE_DIR: str = "./models"
    DEFAULT_DEVICE: str = "cpu"  # "cuda" 或 "cpu"
    MAX_CONCURRENT_TASKS: int = 2
    TASK_TIMEOUT: int = 300  # 5分钟

    # 处理参数默认值
    DEFAULT_BLEND_STRENGTH: float = 0.85
    DEFAULT_EDGE_SMOOTHING: float = 0.7
    DEFAULT_LIGHTING_MATCH: float = 0.6

    # 限流配置
    RATE_LIMIT_UPLOAD_PER_HOUR: int = 20
    RATE_LIMIT_PROCESS_PER_DAY: int = 50

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # CORS配置
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Prometheus配置
    PROMETHEUS_PORT: int = 9090
    ENABLE_METRICS: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()