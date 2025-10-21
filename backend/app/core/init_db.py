from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal, Base
from app.models import *  # 导入所有模型
from app.utils.logger import logger


async def init_db() -> None:
    """初始化数据库表"""
    async with AsyncSessionLocal() as session:
        # 这里可以添加数据库迁移逻辑
        # 目前使用 SQLAlchemy 自动创建表
        from app.core.database import engine

        # 创建所有表
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created successfully")


async def create_default_configs() -> None:
    """创建默认配置"""
    from app.models.system_config import SystemConfig

    default_configs = [
        {
            "config_key": "max_file_size",
            "config_value": 10485760,  # 10MB
            "description": "最大文件上传大小(字节)"
        },
        {
            "config_key": "supported_formats",
            "config_value": ["jpg", "jpeg", "png", "webp"],
            "description": "支持的图片格式"
        },
        {
            "config_key": "default_processing_params",
            "config_value": {
                "blend_strength": 0.85,
                "edge_smoothing": 0.7,
                "lighting_match": 0.6
            },
            "description": "默认处理参数"
        },
        {
            "config_key": "rate_limits",
            "config_value": {
                "upload_per_hour": 20,
                "process_per_day": 50
            },
            "description": "频率限制配置"
        }
    ]

    async with AsyncSessionLocal() as session:
        for config_data in default_configs:
            # 检查配置是否已存在
            existing = await session.get(SystemConfig, config_data["config_key"])
            if not existing:
                config = SystemConfig(**config_data)
                session.add(config)

        await session.commit()
        logger.info("Default configurations created successfully")