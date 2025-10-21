#!/usr/bin/env python3
"""
模型设置脚本
创建必要的模拟模型文件用于开发测试
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        else:
            print(f"⚠️  Mock model already exists: {filename}")

    # 尝试下载OpenCV Haar分类器
    try:
        import requests
        import cv2

        haar_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
        haar_path = models_dir / "haarcascade_frontalface_default.xml"

        if not haar_path.exists():
            print("📥 Downloading OpenCV Haar Cascade...")
            response = requests.get(haar_url, timeout=30)
            response.raise_for_status()

            with open(haar_path, 'wb') as f:
                f.write(response.content)
            print("✅ OpenCV Haar Cascade downloaded successfully")
        else:
            print("⚠️  OpenCV Haar Cascade already exists")

    except Exception as e:
        print(f"❌ Failed to download Haar Cascade: {str(e)}")
        print("   OpenCV will use its built-in cascade instead")

def check_requirements():
    """检查依赖包"""
    required_packages = [
        'torch',
        'opencv-python',
        'numpy',
        'Pillow',
        'fastapi',
        'sqlalchemy'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def setup_directories():
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

def main():
    """主函数"""
    print("🚀 AI Hairstyle Project Setup")
    print("=============================")

    # 检查依赖
    if not check_requirements():
        sys.exit(1)

    # 创建目录
    print("\n📁 Setting up directories...")
    setup_directories()

    # 创建模拟模型
    print("\n🤖 Setting up mock models...")
    create_mock_models()

    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("   1. Copy .env.example to .env and configure if needed")
    print("   2. Run 'python start_dev.py' to start the development server")
    print("   3. Visit http://localhost:8000/api/v1/docs for API documentation")
    print("   4. Visit http://localhost:3000 for the frontend application")

if __name__ == "__main__":
    main()