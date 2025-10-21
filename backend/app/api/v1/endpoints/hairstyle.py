from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.services.hairstyle_service import HairstyleService
from app.schemas.hairstyle import (
    HairstyleTextRequest,
    HairstyleReferenceRequest,
    HairstyleTaskResponse,
    HairstyleResultResponse
)
from app.utils.logger import logger

router = APIRouter()


@router.post("/text-to-hair", response_model=HairstyleTaskResponse)
async def text_to_hair(
    background_tasks: BackgroundTasks,
    request: HairstyleTextRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    文本描述换发型
    """
    try:
        hairstyle_service = HairstyleService(db)
        task = await hairstyle_service.create_text_to_hair_task(request)

        # 异步处理任务
        background_tasks.add_task(hairstyle_service.process_hairstyle_task, task.task_id)

        logger.info(f"Text-to-hair task created: {task.task_id}")
        return task

    except Exception as e:
        logger.error(f"Text-to-hair task creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/reference-to-hair", response_model=HairstyleTaskResponse)
async def reference_to_hair(
    background_tasks: BackgroundTasks,
    target_image: str,  # 目标图片ID
    reference_image: str,  # 参考图片ID
    parameters: dict = None,
    db: AsyncSession = Depends(get_db)
):
    """
    参考图片换发型
    """
    try:
        request = HairstyleReferenceRequest(
            target_image_id=target_image,
            reference_image_id=reference_image,
            parameters=parameters or {}
        )

        hairstyle_service = HairstyleService(db)
        task = await hairstyle_service.create_reference_to_hair_task(request)

        # 异步处理任务
        background_tasks.add_task(hairstyle_service.process_hairstyle_task, task.task_id)

        logger.info(f"Reference-to-hair task created: {task.task_id}")
        return task

    except Exception as e:
        logger.error(f"Reference-to-hair task creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tasks/{task_id}", response_model=HairstyleTaskResponse)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    获取任务状态
    """
    try:
        hairstyle_service = HairstyleService(db)
        task = await hairstyle_service.get_task_status(task_id)
        return task

    except Exception as e:
        logger.error(f"Get task status failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Task not found")


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    取消处理任务
    """
    try:
        hairstyle_service = HairstyleService(db)
        await hairstyle_service.cancel_task(task_id)

        logger.info(f"Task cancelled: {task_id}")
        return {"message": "Task cancelled successfully"}

    except Exception as e:
        logger.error(f"Task cancellation failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Task not found")


@router.get("/results/{result_id}", response_model=HairstyleResultResponse)
async def get_result(
    result_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    获取处理结果
    """
    try:
        hairstyle_service = HairstyleService(db)
        result = await hairstyle_service.get_result(result_id)
        return result

    except Exception as e:
        logger.error(f"Get result failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Result not found")


@router.get("/results/{result_id}/download")
async def download_result(
    result_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    下载结果图片
    """
    try:
        hairstyle_service = HairstyleService(db)
        download_url = await hairstyle_service.get_download_url(result_id)

        return {"download_url": download_url}

    except Exception as e:
        logger.error(f"Download result failed: {str(e)}")
        raise HTTPException(status_code=404, detail="Result not found")


@router.get("/history", response_model=List[HairstyleResultResponse])
async def get_history(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    获取历史记录
    """
    try:
        hairstyle_service = HairstyleService(db)
        results = await hairstyle_service.get_history(skip=skip, limit=limit)
        return results

    except Exception as e:
        logger.error(f"Get history failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.websocket("/ws/hairstyle/{task_id}")
async def websocket_task_status(
    websocket: WebSocket,
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    实时任务状态推送
    """
    await websocket.accept()

    try:
        hairstyle_service = HairstyleService(db)

        # 获取任务状态
        task = await hairstyle_service.get_task_status(task_id)

        # 如果任务已完成，直接发送结果
        if task.status in ["completed", "failed"]:
            await websocket.send_json({
                "type": task.status,
                "data": {
                    "task_id": task_id,
                    "progress": 100,
                    "message": "Task completed" if task.status == "completed" else "Task failed",
                    "error": task.error_message if task.status == "failed" else None
                }
            })
            await websocket.close()
            return

        # 实时推送任务状态
        async for message in hairstyle_service.watch_task_status(task_id):
            await websocket.send_json(message)

            if message.get("type") in ["completed", "failed"]:
                await websocket.close()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task: {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {str(e)}")
        await websocket.close(code=1011)