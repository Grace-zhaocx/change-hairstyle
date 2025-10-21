#!/usr/bin/env python3
"""
开发环境启动脚本
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from app.core.init_db import init_db
from app.core.config import settings
from app.utils.logger import logger


async def setup_dev_environment():
    """设置开发环境"""
    try:
        # 创建必要的目录
        directories = [
            settings.UPLOAD_DIR,
            settings.MODEL_CACHE_DIR,
            "data/results",
            "logs",
            "static/uploads"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")

        # 初始化数据库
        await init_db()
        logger.info("Database initialized successfully")

        logger.info("Development environment setup completed!")
        return True

    except Exception as e:
        logger.error(f"Failed to setup development environment: {str(e)}")
        return False


if __name__ == "__main__":
    print("Setting up AI Hairstyle API development environment...")

    # 运行环境设置
    success = asyncio.run(setup_dev_environment())

    if success:
        print("\n✅ Development environment ready!")
        print("\n🚀 Starting API server...")
        print("📖 API Documentation: http://localhost:8000/api/v1/docs")
        print("🌐 API Root: http://localhost:8000")
        print("\nPress Ctrl+C to stop the server")

        # 启动API服务器
        import uvicorn
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level=settings.LOG_LEVEL.lower()
        )
    else:
        print("\n❌ Failed to setup development environment")
        sys.exit(1)