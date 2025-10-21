#!/usr/bin/env python3
"""
简化的启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def create_directories():
    """创建必要的目录"""
    directories = [
        "uploads",
        "data",
        "data/results",
        "logs",
        "static/uploads",
        "models"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def create_mock_models():
    """创建模拟模型文件"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    mock_files = {
        "retinaface_mobilenet.pth": b"MOCK_RETINAFACE_WEIGHTS",
        "segformer_hair.pth": b"MOCK_SEGFORMER_WEIGHTS",
        "stable_diffusion_v1_5.safetensors": b"MOCK_STABLE_DIFFUSION_WEIGHTS",
    }

    for filename, content in mock_files.items():
        file_path = models_dir / filename
        if not file_path.exists():
            with open(file_path, 'wb') as f:
                f.write(content)
            print(f"✅ Created mock model: {filename}")

def create_env_file():
    """创建环境变量文件"""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# AI智能换发型网站 - 开发环境配置

# 基础配置
PROJECT_NAME=AI Hairstyle API
VERSION=1.0.0
API_V1_STR=/api/v1
SECRET_KEY=dev-secret-key-change-in-production-12345

# 环境配置
ENVIRONMENT=development
DEBUG=true

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 数据库配置
DATABASE_URL=sqlite+aiosqlite:///./hairstyle_dev.db

# 文件上传配置
MAX_FILE_SIZE=10485760
ALLOWED_IMAGE_TYPES=["image/jpeg", "image/png", "image/webp"]
UPLOAD_DIR=uploads

# AI模型配置
MODEL_CACHE_DIR=./models
DEFAULT_DEVICE=cpu

# 日志配置
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")

def main():
    """主函数"""
    print("🚀 AI智能换发型网站 - 简化启动")
    print("===============================")

    # 创建目录
    print("\n📁 创建目录...")
    create_directories()

    # 创建环境文件
    print("\n📝 创建配置文件...")
    create_env_file()

    # 创建模拟模型
    print("\n🤖 创建模拟模型...")
    create_mock_models()

    print("\n✅ 基础环境准备完成!")
    print("\n🎯 现在可以启动服务器:")
    print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("\n📖 访问地址:")
    print("   前端应用: http://localhost:3000")
    print("   API文档:  http://localhost:8000/api/v1/docs")

if __name__ == "__main__":
    main()