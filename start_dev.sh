#!/bin/bash

# AI智能换发型网站 - 开发环境快速启动脚本

echo "🚀 AI智能换发型网站 - 开发环境启动"
echo "=================================="

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装，请先安装Python 3.9+"
    exit 1
fi

# 检查Node.js环境
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装Node.js 18+"
    exit 1
fi

echo "✅ 环境检查通过"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装Python依赖
echo "📥 安装Python依赖..."
cd backend
pip install -r requirements.txt

# 创建环境变量文件
if [ ! -f ".env" ]; then
    echo "📝 创建环境变量配置..."
    cp .env.example .env
fi

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p uploads data results logs static models

# 启动后端服务（后台运行）
echo "🖥️  启动后端服务..."
python start_dev.py &
BACKEND_PID=$!

cd ..

# 安装Node.js依赖
echo "📥 安装Node.js依赖..."
cd frontend
npm install

# 启动前端服务（后台运行）
echo "🌐 启动前端服务..."
npm run dev &
FRONTEND_PID=$!

cd ..

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
echo "🔍 检查服务状态..."

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 后端服务启动成功"
    echo "📖 API文档: http://localhost:8000/api/v1/docs"
else
    echo "❌ 后端服务启动失败"
fi

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ 前端服务启动成功"
    echo "🌐 前端页面: http://localhost:3000"
else
    echo "⏳ 前端服务正在启动中..."
    echo "🌐 前端页面: http://localhost:3000"
fi

echo ""
echo "🎉 开发环境启动完成！"
echo ""
echo "📋 访问地址："
echo "   前端应用: http://localhost:3000"
echo "   API文档:  http://localhost:8000/api/v1/docs"
echo "   API根路径: http://localhost:8000"
echo ""
echo "🛑 停止服务："
echo "   按 Ctrl+C 停止脚本，或运行以下命令："
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "📚 更多信息请查看 README_DEVELOPMENT.md"

# 等待用户中断
trap 'echo ""; echo "🛑 正在停止服务..."; kill $BACKEND_PID $FRONTEND_PID; echo "✅ 服务已停止"; exit 0' INT

echo "按 Ctrl+C 停止所有服务"
while true; do
    sleep 1
done