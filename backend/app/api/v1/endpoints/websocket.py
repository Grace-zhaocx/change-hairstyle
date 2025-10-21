"""
WebSocket API端点
用于实时任务进度推送
"""

import json
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.hairstyle_service import HairstyleService
from app.websocket.ws_manager import manager, get_websocket_manager
from app.utils.logger import logger

router = APIRouter()


@router.websocket("/hairstyle/{task_id}")
async def websocket_hairstyle_progress(
    websocket: WebSocket,
    task_id: str,
    db: AsyncSession = Depends(get_db)
):
    """发型处理进度WebSocket端点"""
    connection_id = None
    try:
        # 建立WebSocket连接
        connection_id = await manager.connect(websocket, task_id)

        # 获取任务当前状态
        hairstyle_service = HairstyleService(db)
        try:
            task_status = await hairstyle_service.get_task_status(task_id)

            # 发送当前状态
            await manager.send_to_connection(connection_id, {
                'type': 'status_update',
                'data': {
                    'task_id': task_id,
                    'status': task_status.status,
                    'progress': task_status.progress,
                    'current_stage': task_status.current_stage,
                    'error_message': task_status.error_message,
                    'estimated_time': None
                }
            })

            logger.info(f"Sent initial status for task {task_id} to connection {connection_id}")

        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {str(e)}")
            await manager.send_to_connection(connection_id, {
                'type': 'error',
                'data': {
                    'message': 'Failed to get task status',
                    'error': str(e)
                }
            })

        # 保持连接活跃，监听客户端消息
        while True:
            try:
                # 等待客户端消息（心跳等）
                data = await websocket.receive_text()
                message = json.loads(data)

                # 处理客户端消息
                if message.get('type') == 'ping':
                    await manager.send_to_connection(connection_id, {
                        'type': 'pong',
                        'data': {
                            'timestamp': asyncio.get_event_loop().time()
                        }
                    })
                elif message.get('type') == 'get_status':
                    # 重新获取任务状态
                    try:
                        task_status = await hairstyle_service.get_task_status(task_id)
                        await manager.send_to_connection(connection_id, {
                            'type': 'status_update',
                            'data': {
                                'task_id': task_id,
                                'status': task_status.status,
                                'progress': task_status.progress,
                                'current_stage': task_status.current_stage,
                                'error_message': task_status.error_message
                            }
                        })
                    except Exception as e:
                        logger.error(f"Failed to refresh task status: {str(e)}")

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from connection {connection_id}")
            except Exception as e:
                logger.error(f"Error handling message from {connection_id}: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {str(e)}")
    finally:
        # 清理连接
        if connection_id:
            manager.disconnect(connection_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """获取WebSocket连接统计信息"""
    return {
        "total_connections": manager.get_total_connections(),
        "active_tasks": len(manager.active_connections),
        "connections_by_task": {
            task_id: len(connections)
            for task_id, connections in manager.active_connections.items()
        }
    }


# WebSocket消息发送工具函数
async def send_task_progress_update(task_id: str, progress: int, stage: str, message: str = None):
    """发送任务进度更新"""
    await manager.send_progress_update(task_id, progress, stage, message)


async def send_task_status_update(task_id: str, status: str, message: str = None):
    """发送任务状态更新"""
    await manager.send_status_update(task_id, status, message)


async def send_task_completed(task_id: str, result_data: Dict[str, Any]):
    """发送任务完成通知"""
    await manager.send_task_completed(task_id, result_data)


async def send_task_failed(task_id: str, error: str, details: Dict[str, Any] = None):
    """发送任务失败通知"""
    await manager.send_task_failed(task_id, error, details)