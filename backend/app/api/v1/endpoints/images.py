from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.services.image_service import ImageService
from app.schemas.image import ImageUploadResponse, ImageInfo
from app.utils.logger import logger

router = APIRouter()


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    上传图片
    """
    try:
        image_service = ImageService(db)
        result = await image_service.upload_image(file)

        # 异步处理人脸检测
        background_tasks.add_task(image_service.detect_face, result.image_id)

        logger.info(f"Image uploaded successfully: {result.image_id}")
        return result

    except Exception as e:
        logger.error(f"Image upload failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{image_id}", response_model=ImageInfo)
async def get_image_info(
    image_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    获取图片信息
    """
    try:
        image_service = ImageService(db)
        result = await image_service.get_image_info(image_id)
        return result

    except Exception as e:
        logger.error(f"Get image info failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Image not found")


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    删除图片
    """
    try:
        image_service = ImageService(db)
        await image_service.delete_image(image_id)

        logger.info(f"Image deleted successfully: {image_id}")
        return {"message": "Image deleted successfully"}

    except Exception as e:
        logger.error(f"Image deletion failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Image not found")


@router.get("/", response_model=List[ImageInfo])
async def list_images(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    获取图片列表
    """
    try:
        image_service = ImageService(db)
        results = await image_service.list_images(skip=skip, limit=limit)
        return results

    except Exception as e:
        logger.error(f"List images failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")