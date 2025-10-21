# AI智能换发型网站

基于AI技术的在线换发型网站，用户可以上传个人头像，通过文本描述更换发型，同时保持面部特征不变。

## 项目结构

```
change-hairstyle/
├── frontend/              # React前端应用
├── backend/               # FastAPI后端服务
├── docs/                  # 项目文档
├── scripts/               # 部署和工具脚本
├── models/                # AI模型文件存储
├── data/                  # 数据存储目录
├── logs/                  # 日志文件
├── config/                # 配置文件
├── nginx/                 # Nginx配置
├── ssl/                   # SSL证书
├── docker-compose.yml     # Docker编排文件
├── .env.example           # 环境变量示例
└── README.md              # 项目说明
```

## 快速开始

### 环境要求

- Node.js 18+
- Python 3.9+
- Docker & Docker Compose
- NVIDIA GPU (可选，用于AI加速)

### 开发环境启动

1. 克隆项目
```bash
git clone <repository-url>
cd change-hairstyle
```

2. 启动开发环境
```bash
# 复制环境变量文件
cp .env.example .env

# 启动所有服务
docker-compose up -d

# 或者分别启动前后端
# 启动后端
cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 启动前端
cd frontend && npm run dev
```

3. 访问应用
- 前端界面: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

## 核心功能

- 📸 **图片上传**: 支持JPG、PNG、WebP格式
- 🎯 **文本描述换发型**: 通过自然语言描述想要的发型
- 📱 **实时预览**: 即时查看换发型效果
- ⚙️ **参数调整**: 微调融合度、边缘过渡等参数
- 💾 **结果保存**: 下载高质量效果图

## 技术栈

### 前端
- React 18 + TypeScript
- Ant Design UI组件库
- Redux Toolkit状态管理
- Vite构建工具

### 后端
- FastAPI + Python
- SQLAlchemy ORM
- PostgreSQL数据库
- Redis缓存

### AI模型
- RetinaFace人脸检测
- SegFormer头发分割
- Stable Diffusion发型生成

## 开发指南

详细的开发文档请参考 `docs/` 目录：
- [需求文档](./换发型网站需求文档.md)
- [原型设计](./ASCII原型设计.md)
- [技术设计](./技术设计文档.md)

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**注意**: 本项目仅用于学习和研究目的，请勿用于商业用途。使用时请遵守相关法律法规和用户隐私保护要求。