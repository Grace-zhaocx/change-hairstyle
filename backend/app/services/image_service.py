import os
import uuid
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import UploadFile, HTTPException
from PIL import Image as PILImage
import aiofiles

from app.models.image import Image as ImageModel
from app.schemas.image import ImageUploadResponse, ImageInfo
from app.core.config import settings
from app.utils.logger import logger


class ImageService:
    """图片处理服务"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)

    async def upload_image(self, file: UploadFile) -> ImageUploadResponse:
        """上传图片"""
        try:
            # 验证文件类型
            if not self._is_valid_image_type(file.content_type):
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的文件类型: {file.content_type}"
                )

            # 验证文件大小
            file_content = await file.read()
            if len(file_content) > settings.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE} bytes"
                )

            # 生成文件信息
            file_id = str(uuid.uuid4())
            original_filename = file.filename or "unknown"
            file_extension = Path(original_filename).suffix.lower()
            new_filename = f"{file_id}{file_extension}"
            file_path = self.upload_dir / new_filename

            # 保存文件
            await self._save_file(file_path, file_content)

            # 计算文件哈希
            md5_hash = hashlib.md5(file_content).hexdigest()

            # 获取图片信息
            try:
                with PILImage.open(file_path) as img:
                    width, height = img.size
                    format_name = img.format.lower()
            except Exception as e:
                logger.error(f"Failed to open image {file_path}: {str(e)}")
                raise HTTPException(status_code=400, detail="无效的图片文件")

            # 创建数据库记录
            db_image = ImageModel(
                id=file_id,
                user_id="anonymous",  # 暂时使用匿名用户，后续添加用户系统
                original_filename=original_filename,
                file_path=str(file_path),
                file_size=len(file_content),
                width=width,
                height=height,
                format=format_name,
                mime_type=file.content_type,
                md5_hash=md5_hash,
                face_detected=False,
                status="uploaded"
            )

            self.db.add(db_image)
            await self.db.commit()
            await self.db.refresh(db_image)

            # 构建响应
            response = ImageUploadResponse(
                image_id=db_image.id,
                file_url=f"/static/uploads/{new_filename}",
                thumbnail_url=f"/static/uploads/thumbnails/{new_filename}",  # 稍后实现缩略图
                filename=original_filename,
                file_size=len(file_content),
                width=width,
                height=height,
                format=format_name,
                mime_type=file.content_type,
                md5_hash=md5_hash,
                face_detected=False,
                face_bbox=None,
                upload_time=db_image.created_at
            )

            logger.info(f"Image uploaded successfully: {file_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to upload image: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="图片上传失败")

    async def get_image_info(self, image_id: str) -> ImageInfo:
        """获取图片信息"""
        try:
            result = await self.db.execute(
                select(ImageModel).where(ImageModel.id == image_id)
            )
            image = result.scalar_one_or_none()

            if not image:
                raise HTTPException(status_code=404, detail="图片不存在")

            return ImageInfo(
                id=image.id,
                user_id=image.user_id,
                original_filename=image.original_filename,
                file_path=image.file_path,
                file_size=image.file_size,
                width=image.width,
                height=image.height,
                format=image.format,
                mime_type=image.mime_type,
                md5_hash=image.md5_hash,
                face_detected=image.face_detected,
                face_bbox=image.face_bbox,
                status=image.status,
                created_at=image.created_at,
                updated_at=image.updated_at
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get image info: {str(e)}")
            raise HTTPException(status_code=500, detail="获取图片信息失败")

    async def list_images(self, skip: int = 0, limit: int = 20) -> List[ImageInfo]:
        """获取图片列表"""
        try:
            result = await self.db.execute(
                select(ImageModel)
                .offset(skip)
                .limit(limit)
                .order_by(ImageModel.created_at.desc())
            )
            images = result.scalars().all()

            return [
                ImageInfo(
                    id=image.id,
                    user_id=image.user_id,
                    original_filename=image.original_filename,
                    file_path=image.file_path,
                    file_size=image.file_size,
                    width=image.width,
                    height=image.height,
                    format=image.format,
                    mime_type=image.mime_type,
                    md5_hash=image.md5_hash,
                    face_detected=image.face_detected,
                    face_bbox=image.face_bbox,
                    status=image.status,
                    created_at=image.created_at,
                    updated_at=image.updated_at
                )
                for image in images
            ]

        except Exception as e:
            logger.error(f"Failed to list images: {str(e)}")
            raise HTTPException(status_code=500, detail="获取图片列表失败")

    async def delete_image(self, image_id: str) -> bool:
        """删除图片"""
        try:
            result = await self.db.execute(
                select(ImageModel).where(ImageModel.id == image_id)
            )
            image = result.scalar_one_or_none()

            if not image:
                raise HTTPException(status_code=404, detail="图片不存在")

            # 删除文件
            file_path = Path(image.file_path)
            if file_path.exists():
                file_path.unlink()

            # 删除数据库记录
            await self.db.delete(image)
            await self.db.commit()

            logger.info(f"Image deleted successfully: {image_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete image: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="删除图片失败")

    async def detect_face(self, image_id: str) -> Dict[str, Any]:
        """检测人脸"""
        try:
            # 获取图片信息
            result = await self.db.execute(
                select(ImageModel).where(ImageModel.id == image_id)
            )
            image = result.scalar_one_or_none()

            if not image:
                raise HTTPException(status_code=404, detail="图片不存在")

            # 使用AI服务进行人脸检测
            from app.services.ai_service import AIService
            ai_service = AIService()

            detection_result = await ai_service.detect_faces(image.file_path)

            if detection_result["faces_detected"] > 0:
                # 获取第一个人脸的边界框
                face_data = detection_result["faces"][0]
                bbox = face_data["bbox"]

                face_bbox = {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3],
                    "confidence": face_data.get("confidence", 0.0),
                    "landmarks": face_data.get("landmarks", [])
                }

                # 更新数据库
                image.face_detected = True
                image.face_bbox = face_bbox
                await self.db.commit()

                logger.info(f"Face detected for image: {image_id}")
                return {
                    "detected": True,
                    "face_count": detection_result["faces_detected"],
                    "face_bbox": face_bbox,
                    "detection_result": detection_result
                }
            else:
                # 没有检测到人脸
                image.face_detected = False
                image.face_bbox = None
                await self.db.commit()

                logger.info(f"No face detected for image: {image_id}")
                return {
                    "detected": False,
                    "face_count": 0,
                    "detection_result": detection_result
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to detect face: {str(e)}")
            return {"detected": False, "error": str(e)}

    def _is_valid_image_type(self, content_type: str) -> bool:
        """验证图片类型"""
        return content_type in settings.ALLOWED_IMAGE_TYPES

    async def _save_file(self, file_path: Path, content: bytes) -> None:
        """保存文件"""
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

    async def _create_thumbnail(self, image_path: Path, thumbnail_path: Path, size: tuple = (200, 200)) -> None:
        """创建缩略图"""
        try:
            with PILImage.open(image_path) as img:
                img.thumbnail(size, PILImage.Resampling.LANCZOS)
                img.save(thumbnail_path, optimize=True, quality=85)
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {str(e)}")

    async def _validate_image_content(self, file_path: Path) -> bool:
        """验证图片内容"""
        try:
            with PILImage.open(file_path) as img:
                img.verify()  # 验证图片完整性
            return True
        except Exception:
            return False