from celery import Celery
from app.core.config import settings
from app.services.hairstyle_service import HairstyleService
from app.core.database import AsyncSessionLocal
from app.utils.logger import logger

# 创建Celery实例
celery_app = Celery(
    "hairstyle_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker"]
)

# Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.TASK_TIMEOUT,  # 5分钟超时
    task_soft_time_limit=settings.TASK_TIMEOUT - 60,  # 4分钟软超时
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


@celery_app.task(bind=True)
def process_hairstyle_task(self, task_id: str):
    """处理发型任务"""
    logger.info(f"Starting Celery task: {task_id}")

    try:
        import asyncio

        async def process_task():
            async with AsyncSessionLocal() as db:
                service = HairstyleService(db)
                await service.process_hairstyle_task(task_id)

        # 运行异步任务
        asyncio.run(process_task())

        logger.info(f"Celery task completed: {task_id}")
        return {"status": "completed", "task_id": task_id}

    except Exception as e:
        logger.error(f"Celery task failed: {task_id}, error: {str(e)}")
        return {"status": "failed", "task_id": task_id, "error": str(e)}


# 健康检查任务
@celery_app.task
def health_check():
    """健康检查任务"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


if __name__ == "__main__":
    celery_app.start()