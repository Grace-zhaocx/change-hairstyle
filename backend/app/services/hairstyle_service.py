import asyncio
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException

from app.models.hairstyle_task import HairstyleTask
from app.models.hairstyle_result import HairstyleResult
from app.models.image import Image
from app.schemas.hairstyle import (
    HairstyleTextRequest,
    HairstyleReferenceRequest,
    HairstyleTaskResponse,
    HairstyleResultResponse,
    TaskStatus
)
from app.services.ai_service import AIService
from app.utils.logger import logger


class HairstyleService:
    """发型处理服务"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.ai_service = AIService()

    async def create_text_to_hair_task(self, request: HairstyleTextRequest) -> HairstyleTaskResponse:
        """创建文本描述换发型任务"""
        try:
            # 验证源图片
            image_result = await self.db.execute(
                select(Image).where(Image.id == request.image_id)
            )
            source_image = image_result.scalar_one_or_none()

            if not source_image:
                raise HTTPException(status_code=404, detail="源图片不存在")

            # 创建任务
            task_id = str(uuid.uuid4())
            db_task = HairstyleTask(
                id=task_id,
                user_id="anonymous",  # 暂时使用匿名用户
                source_image_id=request.image_id,
                task_type="text",
                input_params={
                    "description": request.description.dict(),
                    "parameters": request.parameters.dict() if request.parameters else {}
                },
                status=TaskStatus.PENDING,
                progress=0
            )

            self.db.add(db_task)
            await self.db.commit()
            await self.db.refresh(db_task)

            logger.info(f"Text-to-hair task created: {task_id}")
            return self._task_to_response(db_task)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create text-to-hair task: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="创建任务失败")

    async def create_reference_to_hair_task(self, request: HairstyleReferenceRequest) -> HairstyleTaskResponse:
        """创建参考图片换发型任务"""
        try:
            # 验证目标图片
            target_result = await self.db.execute(
                select(Image).where(Image.id == request.target_image_id)
            )
            target_image = target_result.scalar_one_or_none()

            if not target_image:
                raise HTTPException(status_code=404, detail="目标图片不存在")

            # 验证参考图片
            ref_result = await self.db.execute(
                select(Image).where(Image.id == request.reference_image_id)
            )
            reference_image = ref_result.scalar_one_or_none()

            if not reference_image:
                raise HTTPException(status_code=404, detail="参考图片不存在")

            # 创建任务
            task_id = str(uuid.uuid4())
            db_task = HairstyleTask(
                id=task_id,
                user_id="anonymous",
                source_image_id=request.target_image_id,
                reference_image_id=request.reference_image_id,
                task_type="reference",
                input_params={
                    "style_similarity": request.style_similarity,
                    "parameters": request.parameters.dict() if request.parameters else {}
                },
                status=TaskStatus.PENDING,
                progress=0
            )

            self.db.add(db_task)
            await self.db.commit()
            await self.db.refresh(db_task)

            logger.info(f"Reference-to-hair task created: {task_id}")
            return self._task_to_response(db_task)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create reference-to-hair task: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="创建任务失败")

    async def get_task_status(self, task_id: str) -> HairstyleTaskResponse:
        """获取任务状态"""
        try:
            result = await self.db.execute(
                select(HairstyleTask).where(HairstyleTask.id == task_id)
            )
            task = result.scalar_one_or_none()

            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")

            return self._task_to_response(task)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task status: {str(e)}")
            raise HTTPException(status_code=500, detail="获取任务状态失败")

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        try:
            result = await self.db.execute(
                select(HairstyleTask).where(HairstyleTask.id == task_id)
            )
            task = result.scalar_one_or_none()

            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")

            if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                raise HTTPException(status_code=400, detail="任务无法取消")

            # 更新任务状态
            task.status = TaskStatus.CANCELLED
            task.updated_at = datetime.utcnow()
            if task.status == TaskStatus.PROCESSING:
                task.completed_at = datetime.utcnow()

            await self.db.commit()

            logger.info(f"Task cancelled: {task_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel task: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="取消任务失败")

    async def get_result(self, result_id: str) -> HairstyleResultResponse:
        """获取处理结果"""
        try:
            result = await self.db.execute(
                select(HairstyleResult).where(HairstyleResult.id == result_id)
            )
            db_result = result.scalar_one_or_none()

            if not db_result:
                raise HTTPException(status_code=404, detail="结果不存在")

            return self._result_to_response(db_result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get result: {str(e)}")
            raise HTTPException(status_code=500, detail="获取结果失败")

    async def get_download_url(self, result_id: str) -> str:
        """获取下载链接"""
        try:
            result = await self.db.execute(
                select(HairstyleResult).where(HairstyleResult.id == result_id)
            )
            db_result = result.scalar_one_or_none()

            if not db_result:
                raise HTTPException(status_code=404, detail="结果不存在")

            # 获取结果图片信息
            image_result = await self.db.execute(
                select(Image).where(Image.id == db_result.result_image_id)
            )
            result_image = image_result.scalar_one_or_none()

            if not result_image:
                raise HTTPException(status_code=404, detail="结果图片不存在")

            # 增加下载计数
            db_result.download_count += 1
            await self.db.commit()

            # 返回下载URL
            download_url = f"/static/uploads/{result_image.file_path.split('/')[-1]}"
            return download_url

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get download URL: {str(e)}")
            raise HTTPException(status_code=500, detail="获取下载链接失败")

    async def get_history(self, skip: int = 0, limit: int = 20) -> List[HairstyleResultResponse]:
        """获取历史记录"""
        try:
            result = await self.db.execute(
                select(HairstyleResult)
                .order_by(HairstyleResult.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            results = result.scalars().all()

            return [self._result_to_response(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to get history: {str(e)}")
            raise HTTPException(status_code=500, detail="获取历史记录失败")

    async def process_hairstyle_task(self, task_id: str):
        """处理发型任务（异步）"""
        try:
            # 获取任务
            task_result = await self.db.execute(
                select(HairstyleTask).where(HairstyleTask.id == task_id)
            )
            task = task_result.scalar_one_or_none()

            if not task:
                logger.error(f"Task not found: {task_id}")
                return

            # 更新任务状态为处理中
            task.status = TaskStatus.PROCESSING
            task.current_stage = "initializing"
            task.started_at = datetime.utcnow()
            await self.db.commit()

            logger.info(f"Starting hairstyle task processing: {task_id}")

            # 获取源图片
            image_result = await self.db.execute(
                select(Image).where(Image.id == task.source_image_id)
            )
            source_image = image_result.scalar_one_or_none()

            if not source_image:
                await self._fail_task(task, "源图片不存在")
                return

            # 处理任务
            if task.task_type == "text":
                await self._process_text_task(task, source_image)
            elif task.task_type == "reference":
                await self._process_reference_task(task, source_image)
            else:
                await self._fail_task(task, f"不支持的任务类型: {task.task_type}")

        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}")
            # 更新任务状态为失败
            task_result = await self.db.execute(
                select(HairstyleTask).where(HairstyleTask.id == task_id)
            )
            task = task_result.scalar_one_or_none()
            if task:
                await self._fail_task(task, str(e))

    async def _process_text_task(self, task: HairstyleTask, source_image: Image):
        """处理文本描述任务"""
        try:
            description = task.input_params.get("description", {})
            parameters = task.input_params.get("parameters", {})

            # 更新进度
            await self._update_progress(task, 20, "face_detection")

            # AI处理
            ai_result = await self.ai_service.process_hairstyle_change(
                source_image.file_path,
                description,
                parameters
            )

            if not ai_result["success"]:
                await self._fail_task(task, ai_result["error"])
                return

            # 创建结果记录
            await self._create_result(task, source_image, ai_result)
            await self._complete_task(task)

        except Exception as e:
            logger.error(f"Text task processing failed: {str(e)}")
            await self._fail_task(task, str(e))

    async def _process_reference_task(self, task: HairstyleTask, source_image: Image):
        """处理参考图片任务"""
        try:
            # TODO: 实现参考图片处理逻辑
            await self._update_progress(task, 50, "reference_analysis")

            # 暂时使用文本处理逻辑
            await self._process_text_task(task, source_image)

        except Exception as e:
            logger.error(f"Reference task processing failed: {str(e)}")
            await self._fail_task(task, str(e))

    async def _update_progress(self, task: HairstyleTask, progress: int, stage: str, message: str = None):
        """更新任务进度"""
        task.progress = progress
        task.current_stage = stage
        task.updated_at = datetime.utcnow()
        await self.db.commit()

        # 发送WebSocket进度更新
        try:
            from app.api.v1.endpoints.websocket import send_task_progress_update
            await send_task_progress_update(
                task_id=str(task.id),
                progress=progress,
                stage=stage,
                message=message or f"正在执行: {stage}"
            )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket progress update: {str(e)}")

    async def _create_result(self, task: HairstyleTask, source_image: Image, ai_result: Dict[str, Any]):
        """创建结果记录"""
        # 创建结果图片记录
        result_image = Image(
            id=str(uuid.uuid4()),
            user_id=task.user_id,
            original_filename=f"result_{task.id}.jpg",
            file_path=ai_result["result_path"],
            file_size=0,  # TODO: 获取实际文件大小
            width=source_image.width,
            height=source_image.height,
            format="jpeg",
            mime_type="image/jpeg",
            md5_hash="",  # TODO: 计算实际哈希
            face_detected=source_image.face_detected,
            face_bbox=source_image.face_bbox,
            status="completed"
        )

        self.db.add(result_image)
        await self.db.flush()

        # 创建发型结果记录
        result = HairstyleResult(
            id=str(uuid.uuid4()),
            task_id=task.id,
            user_id=task.user_id,
            source_image_id=task.source_image_id,
            result_image_id=result_image.id,
            result_params=task.input_params.get("parameters", {}),
            quality_score=0.85,  # TODO: 计算实际质量分数
            download_count=0,
            view_count=0,
            is_public=False
        )

        self.db.add(result)
        await self.db.flush()

        # 更新任务结果ID
        task.result_id = result.id
        await self.db.commit()

        logger.info(f"Result created for task: {task.id}")

    async def _complete_task(self, task: HairstyleTask):
        """完成任务"""
        task.status = TaskStatus.COMPLETED
        task.progress = 100
        task.current_stage = "completed"
        task.completed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        await self.db.commit()

        # 发送WebSocket完成通知
        try:
            from app.api.v1.endpoints.websocket import send_task_completed
            await send_task_completed(
                task_id=str(task.id),
                result_data={
                    "message": "发型处理完成",
                    "completed_at": task.completed_at.isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket completion notification: {str(e)}")

        logger.info(f"Task completed: {task.id}")

    async def _fail_task(self, task: HairstyleTask, error_message: str):
        """任务失败"""
        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.completed_at = datetime.utcnow()
        task.updated_at = datetime.utcnow()
        await self.db.commit()

        # 发送WebSocket失败通知
        try:
            from app.api.v1.endpoints.websocket import send_task_failed
            await send_task_failed(
                task_id=str(task.id),
                error=error_message,
                details={
                    "failed_at": task.completed_at.isoformat(),
                    "stage": task.current_stage
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket failure notification: {str(e)}")

        logger.error(f"Task failed: {task.id}, error: {error_message}")

    def _task_to_response(self, task: HairstyleTask) -> HairstyleTaskResponse:
        """转换任务为响应格式"""
        return HairstyleTaskResponse(
            task_id=str(task.id),
            user_id=str(task.user_id),
            source_image_id=str(task.source_image_id),
            reference_image_id=str(task.reference_image_id) if task.reference_image_id else None,
            task_type=task.task_type,
            input_params=task.input_params,
            status=task.status,
            progress=task.progress,
            current_stage=task.current_stage,
            error_message=task.error_message,
            estimated_time=None,  # TODO: 计算预计时间
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at
        )

    def _result_to_response(self, result: HairstyleResult) -> HairstyleResultResponse:
        """转换结果为响应格式"""
        return HairstyleResultResponse(
            result_id=str(result.id),
            task_id=str(result.task_id),
            user_id=str(result.user_id),
            source_image_url=f"/static/uploads/{result.source_image_id}.jpg",  # TODO: 获取实际URL
            result_image_url=f"/static/uploads/{result.result_image_id}.jpg",  # TODO: 获取实际URL
            result_params=result.result_params,
            quality_score=float(result.quality_score) if result.quality_score else None,
            user_rating=result.user_rating,
            user_feedback=result.user_feedback,
            download_count=result.download_count,
            view_count=result.view_count,
            is_public=result.is_public,
            created_at=result.created_at,
            processing_time=result.task.processing_time if result.task else None
        )

    async def watch_task_status(self, task_id: str):
        """监听任务状态变化（用于WebSocket）"""
        # TODO: 实现WebSocket状态监听
        # 这里可以使用Redis pub/sub或其他机制
        pass