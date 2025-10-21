# AI智能换发型网站 - 开发指南

## 项目概述

这是一个基于AI技术的在线换发型网站，用户可以上传个人头像，通过文本描述或参考图片来更换发型，同时保持面部特征不变。

## 当前完成的功能

### ✅ 已完成

1. **项目架构搭建**
   - React 18 + TypeScript 前端框架
   - FastAPI + Python 后端框架
   - 完整的数据模型和API接口设计
   - Docker化部署配置

2. **图片上传服务**
   - 支持JPG、PNG、WebP格式
   - 最大10MB文件大小限制
   - 自动人脸检测功能
   - 图片信息存储和管理

3. **人脸检测服务**
   - 基于OpenCV Haar级联分类器
   - 支持模拟检测作为fallback
   - 人脸边界框和关键点检测
   - 异步处理机制

4. **文本描述换发型API**
   - 完整的任务管理系统
   - 支持后台异步处理
   - 进度跟踪和状态更新
   - 错误处理和恢复机制

5. **前端页面**
   - 精美的首页设计
   - 功能完整的上传页面
   - 响应式设计
   - 动画效果和用户交互

6. **开发环境配置**
   - 环境变量配置
   - 数据库初始化脚本
   - 开发服务器启动脚本

## 如何运行项目

### 1. 环境准备

```bash
# Python环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Node.js环境
cd frontend
npm install
cd ..
```

### 2. 配置环境变量

```bash
# 复制环境变量配置文件
cp backend/.env.example backend/.env

# 根据需要修改配置（通常开发环境使用默认配置即可）
```

### 3. 启动后端服务

```bash
cd backend

# 方法1: 使用开发启动脚本（推荐）
python start_dev.py

# 方法2: 使用uvicorn直接启动
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端服务启动后，可以访问：
- API文档: http://localhost:8000/api/v1/docs
- API根路径: http://localhost:8000
- 健康检查: http://localhost:8000/health

### 4. 启动前端服务

```bash
cd frontend
npm run dev
```

前端服务启动后，可以访问：http://localhost:3000

## API接口说明

### 图片管理接口

- `POST /api/v1/images/upload` - 上传图片
- `GET /api/v1/images/{image_id}` - 获取图片信息
- `DELETE /api/v1/images/{image_id}` - 删除图片
- `GET /api/v1/images/` - 获取图片列表

### 发型处理接口

- `POST /api/v1/hairstyle/text-to-hair` - 文本描述换发型
- `POST /api/v1/hairstyle/reference-to-hair` - 参考图片换发型
- `GET /api/v1/hairstyle/tasks/{task_id}` - 获取任务状态
- `DELETE /api/v1/hairstyle/tasks/{task_id}` - 取消任务

## 测试流程

### 1. 测试图片上传

```bash
# 使用curl测试上传
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.jpg"
```

### 2. 测试人脸检测

上传的图片会自动进行人脸检测，检测结果会存储在数据库中。

### 3. 测试文本描述换发型

```bash
# 使用curl测试发型处理
curl -X POST "http://localhost:8000/api/v1/hairstyle/text-to-hair" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "your-image-id",
    "description": {
      "length": "medium",
      "style": "wavy",
      "color": "brown",
      "custom_description": "自然微卷中长发"
    },
    "parameters": {
      "blend_strength": 0.85,
      "edge_smoothing": 0.7,
      "lighting_match": 0.6
    }
  }'
```

## 当前限制和TODO

### 🔄 待完成功能

1. **AI模型增强**
   - 集成更先进的RetinaFace人脸检测
   - 实现SegFormer头发分割
   - 集成Stable Diffusion发型生成
   - 添加ControlNet面部保持

2. **前端功能完善**
   - 发型选择页面
   - 处理进度页面
   - 效果预览页面
   - 结果页面

3. **高级功能**
   - WebSocket实时进度推送
   - 参考图片换发型
   - 用户注册登录系统
   - 历史记录功能
   - 图片分享功能

### ⚠️ 当前限制

1. **AI处理能力**
   - 目前使用的是简化的AI实现
   - 发型生成效果为模拟结果
   - 需要GPU支持才能运行真实的深度学习模型

2. **性能限制**
   - 并发处理数量有限
   - 大图片处理可能较慢
   - 缺乏缓存优化

3. **功能限制**
   - 暂不支持用户注册登录
   - 没有历史记录功能
   - 不支持参考图片换发型

## 开发建议

### 1. 代码结构

- 后端采用分层架构：Controller -> Service -> Repository
- 前端采用组件化设计，使用Ant Design UI库
- 数据库使用SQLAlchemy ORM，支持异步操作

### 2. 开发规范

- 后端使用Python类型提示
- 前端使用TypeScript严格模式
- 所有API都有完整的文档说明
- 使用loguru进行日志记录

### 3. 部署建议

- 开发环境可以使用Docker Compose
- 生产环境建议使用Kubernetes
- 需要配置Nginx反向代理
- 使用Redis作为缓存和消息队列

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码变更
4. 创建Pull Request
5. 等待代码审查

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**注意**: 本项目目前处于开发阶段，仅用于学习和研究目的。