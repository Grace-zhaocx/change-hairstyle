from fastapi import APIRouter

from app.api.v1.endpoints import images, hairstyle, auth, websocket

api_router = APIRouter()

# 注册各个模块的路由
api_router.include_router(auth.router, prefix="/auth", tags=["认证"])
api_router.include_router(images.router, prefix="/images", tags=["图片管理"])
api_router.include_router(hairstyle.router, prefix="/hairstyle", tags=["发型处理"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])