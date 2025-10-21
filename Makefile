# Makefile for AI Hairstyle Project

.PHONY: help install dev prod clean test lint format

# 默认目标
help:
	@echo "AI智能换发型项目 - 可用命令:"
	@echo ""
	@echo "开发环境:"
	@echo "  make install    - 安装所有依赖"
	@echo "  make dev        - 启动开发环境"
	@echo "  make dev-fe     - 仅启动前端开发服务"
	@echo "  make dev-be     - 仅启动后端开发服务"
	@echo ""
	@echo "生产环境:"
	@echo "  make prod       - 启动生产环境"
	@echo "  make build      - 构建生产镜像"
	@echo ""
	@echo "代码质量:"
	@echo "  make test       - 运行测试"
	@echo "  make lint       - 代码检查"
	@echo "  make format     - 代码格式化"
	@echo ""
	@echo "工具命令:"
	@echo "  make clean      - 清理临时文件"
	@echo "  make logs       - 查看日志"
	@echo "  make db-init    - 初始化数据库"

# 安装依赖
install:
	@echo "安装前端依赖..."
	cd frontend && npm install
	@echo "安装后端依赖..."
	cd backend && pip install -r requirements.txt
	@echo "依赖安装完成!"

# 开发环境
dev:
	@echo "启动开发环境..."
	docker-compose -f docker-compose.dev.yml up --build

dev-fe:
	@echo "启动前端开发服务..."
	cd frontend && npm run dev

dev-be:
	@echo "启动后端开发服务..."
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产环境
prod:
	@echo "启动生产环境..."
	docker-compose up -d --build

build:
	@echo "构建生产镜像..."
	docker-compose build

# 代码质量
test:
	@echo "运行前端测试..."
	cd frontend && npm test
	@echo "运行后端测试..."
	cd backend && python -m pytest

lint:
	@echo "检查前端代码..."
	cd frontend && npm run lint
	@echo "检查后端代码..."
	cd backend && flake8 app/ && mypy app/

format:
	@echo "格式化前端代码..."
	cd frontend && npm run format
	@echo "格式化后端代码..."
	cd backend && black app/ && isort app/

# 工具命令
clean:
	@echo "清理临时文件..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@rm -rf frontend/node_modules/.cache
	@rm -rf frontend/dist
	@rm -rf backend/__pycache__
	@rm -rf .pytest_cache
	@echo "清理完成!"

logs:
	@echo "查看应用日志..."
	docker-compose logs -f

db-init:
	@echo "初始化数据库..."
	cd backend && python -c "
import asyncio
from app.core.init_db import init_db, create_default_configs
asyncio.run(init_db())
asyncio.run(create_default_configs())
print('数据库初始化完成!')
"

# 数据库操作
db-migrate:
	@echo "运行数据库迁移..."
	cd backend && alembic upgrade head

db-reset:
	@echo "重置数据库..."
	@rm -f backend/data/hairstyle.db
	@make db-init

# 模型下载
download-models:
	@echo "下载AI模型..."
	cd backend && python scripts/download_models.py

# 快速启动 (用于新环境)
quickstart:
	@echo "快速启动项目..."
	@make install
	@make download-models
	@make db-init
	@echo "项目已准备就绪! 运行 'make dev' 启动开发环境"