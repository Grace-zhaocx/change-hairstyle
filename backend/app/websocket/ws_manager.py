"""
WebSocket管理器
用于实时推送任务进度和状态更新
"""

import asyncio
import json
from typing import Dict, List, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
import uuid
from datetime import datetime

from app.utils.logger import logger


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 存储活跃连接 {task_id: {connection_id: WebSocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # 存储连接元数据
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, task_id: str) -> str:
        """建立WebSocket连接"""
        await websocket.accept()

        connection_id = str(uuid.uuid4())

        # 添加到活跃连接
        if task_id not in self.active_connections:
            self.active_connections[task_id] = {}

        self.active_connections[task_id][connection_id] = websocket

        # 存储连接元数据
        self.connection_metadata[connection_id] = {
            'task_id': task_id,
            'connected_at': datetime.utcnow(),
            'last_ping': datetime.utcnow()
        }

        logger.info(f"WebSocket connected: {connection_id} for task {task_id}")

        # 发送连接确认
        await self.send_to_connection(connection_id, {
            'type': 'connected',
            'data': {
                'connection_id': connection_id,
                'task_id': task_id,
                'message': 'WebSocket connection established'
            }
        })

        return connection_id

    def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id in self.connection_metadata:
            task_id = self.connection_metadata[connection_id]['task_id']

            # 从活跃连接中移除
            if task_id in self.active_connections:
                self.active_connections[task_id].pop(connection_id, None)
                # 如果任务没有其他连接，移除任务
                if not self.active_connections[task_id]:
                    del self.active_connections[task_id]

            # 移除元数据
            del self.connection_metadata[connection_id]

            logger.info(f"WebSocket disconnected: {connection_id} for task {task_id}")

    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """向特定连接发送消息"""
        if connection_id in self.connection_metadata:
            task_id = self.connection_metadata[connection_id]['task_id']

            if (task_id in self.active_connections and
                connection_id in self.active_connections[task_id]):

                websocket = self.active_connections[task_id][connection_id]
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logger.error(f"Failed to send message to {connection_id}: {str(e)}")
                    # 连接可能已断开，清理连接
                    self.disconnect(connection_id)

    async def broadcast_to_task(self, task_id: str, message: Dict[str, Any]):
        """向任务的所有连接广播消息"""
        if task_id in self.active_connections:
            disconnected_connections = []

            for connection_id, websocket in self.active_connections[task_id].items():
                try:
                    await websocket.send_text(json.dumps(message, ensure_ascii=False))
                except Exception as e:
                    logger.error(f"Failed to broadcast to {connection_id}: {str(e)}")
                    disconnected_connections.append(connection_id)

            # 清理断开的连接
            for connection_id in disconnected_connections:
                self.disconnect(connection_id)

            logger.info(f"Broadcasted message to {len(self.active_connections[task_id]) - len(disconnected_connections)} connections for task {task_id}")

    async def send_progress_update(self, task_id: str, progress: int, stage: str, message: str = None, estimated_time: int = None):
        """发送进度更新"""
        progress_message = {
            'type': 'progress_update',
            'data': {
                'task_id': task_id,
                'progress': progress,
                'stage': stage,
                'message': message,
                'estimated_time': estimated_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast_to_task(task_id, progress_message)

    async def send_status_update(self, task_id: str, status: str, message: str = None, error: str = None):
        """发送状态更新"""
        status_message = {
            'type': 'status_update',
            'data': {
                'task_id': task_id,
                'status': status,
                'message': message,
                'error': error,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast_to_task(task_id, status_message)

    async def send_task_completed(self, task_id: str, result_data: Dict[str, Any]):
        """发送任务完成通知"""
        completion_message = {
            'type': 'task_completed',
            'data': {
                'task_id': task_id,
                'result': result_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast_to_task(task_id, completion_message)

    async def send_task_failed(self, task_id: str, error: str, details: Dict[str, Any] = None):
        """发送任务失败通知"""
        failure_message = {
            'type': 'task_failed',
            'data': {
                'task_id': task_id,
                'error': error,
                'details': details,
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        await self.broadcast_to_task(task_id, failure_message)

    def get_connection_count(self, task_id: str) -> int:
        """获取任务的连接数量"""
        return len(self.active_connections.get(task_id, {}))

    def get_total_connections(self) -> int:
        """获取总连接数"""
        return sum(len(connections) for connections in self.active_connections.values())

    async def cleanup_stale_connections(self):
        """清理陈旧连接"""
        current_time = datetime.utcnow()
        stale_connections = []

        for connection_id, metadata in self.connection_metadata.items():
            # 清理超过5分钟没有活动的连接
            if (current_time - metadata['last_ping']).total_seconds() > 300:
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            self.disconnect(connection_id)
            logger.info(f"Cleaned up stale connection: {connection_id}")

        if stale_connections:
            logger.info(f"Cleaned up {len(stale_connections)} stale connections")


# 全局连接管理器实例
manager = ConnectionManager()


async def get_websocket_manager() -> ConnectionManager:
    """获取WebSocket管理器实例"""
    return manager