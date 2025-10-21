from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ImageUploadRequest(BaseModel):
    """图片上传请求"""
    description: Optional[str] = Field(None, max_length=500, description="图片描述")


class ImageUploadResponse(BaseModel):
    """图片上传响应"""
    image_id: str = Field(..., description="图片ID")
    file_url: str = Field(..., description="文件URL")
    thumbnail_url: Optional[str] = Field(None, description="缩略图URL")
    filename: str = Field(..., description="原始文件名")
    file_size: int = Field(..., description="文件大小(字节)")
    width: int = Field(..., description="图片宽度")
    height: int = Field(..., description="图片高度")
    format: str = Field(..., description="图片格式")
    mime_type: str = Field(..., description="MIME类型")
    md5_hash: str = Field(..., description="文件MD5哈希")
    face_detected: bool = Field(..., description="是否检测到人脸")
    face_bbox: Optional[Dict[str, int]] = Field(None, description="人脸边界框 {x, y, width, height}")
    upload_time: datetime = Field(..., description="上传时间")


class ImageInfo(BaseModel):
    """图片信息"""
    id: str = Field(..., description="图片ID")
    user_id: str = Field(..., description="用户ID")
    original_filename: str = Field(..., description="原始文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小(字节)")
    width: int = Field(..., description="图片宽度")
    height: int = Field(..., description="图片高度")
    format: str = Field(..., description="图片格式")
    mime_type: str = Field(..., description="MIME类型")
    md5_hash: str = Field(..., description="文件MD5哈希")
    face_detected: bool = Field(..., description="是否检测到人脸")
    face_bbox: Optional[Dict[str, int]] = Field(None, description="人脸边界框")
    status: str = Field(..., description="图片状态")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    class Config:
        from_attributes = True


class ImageProcessRequest(BaseModel):
    """图片处理请求"""
    image_id: str = Field(..., description="图片ID")
    operations: list = Field(..., description="处理操作列表")


class ImageBatchUploadResponse(BaseModel):
    """批量上传响应"""
    successful: list[ImageUploadResponse] = Field(..., description="成功上传的图片")
    failed: list[Dict[str, Any]] = Field(..., description="失败的图片")
    total_count: int = Field(..., description="总数量")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")