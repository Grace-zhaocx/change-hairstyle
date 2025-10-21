from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class HairstyleLength(str, Enum):
    """发型长度"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class HairstyleStyle(str, Enum):
    """发型风格"""
    STRAIGHT = "straight"
    WAVY = "wavy"
    CURLY = "curly"
    BRAIDED = "braided"
    BUZZED = "buzzed"


class HairstyleColor(str, Enum):
    """发色"""
    NATURAL = "natural"
    BROWN = "brown"
    BLONDE = "blonde"
    BLACK = "black"
    RED = "red"
    CUSTOM = "custom"


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HairstyleDescription(BaseModel):
    """发型描述"""
    length: HairstyleLength = Field(..., description="发型长度")
    style: HairstyleStyle = Field(..., description="发型风格")
    color: HairstyleColor = Field(..., description="发色")
    custom_description: Optional[str] = Field(None, max_length=500, description="自定义描述")

    @validator('custom_description')
    def validate_custom_description(cls, v, values):
        if values.get('color') == HairstyleColor.CUSTOM and not v:
            raise ValueError('自定义发色时必须提供描述')
        return v


class HairstyleParameters(BaseModel):
    """发型处理参数"""
    blend_strength: float = Field(0.85, ge=0.0, le=1.0, description="融合强度")
    edge_smoothing: float = Field(0.7, ge=0.0, le=1.0, description="边缘平滑程度")
    lighting_match: float = Field(0.6, ge=0.0, le=1.0, description="光照匹配程度")
    color_intensity: float = Field(1.0, ge=0.0, le=2.0, description="颜色强度")
    detail_preservation: float = Field(0.8, ge=0.0, le=1.0, description="细节保持程度")

    class Config:
        extra = "allow"  # 允许额外参数


class HairstyleTextRequest(BaseModel):
    """文本描述换发型请求"""
    image_id: str = Field(..., description="源图片ID")
    description: HairstyleDescription = Field(..., description="发型描述")
    parameters: Optional[HairstyleParameters] = Field(None, description="处理参数")


class HairstyleReferenceRequest(BaseModel):
    """参考图片换发型请求"""
    target_image_id: str = Field(..., description="目标图片ID")
    reference_image_id: str = Field(..., description="参考图片ID")
    parameters: Optional[HairstyleParameters] = Field(None, description="处理参数")
    style_similarity: float = Field(0.8, ge=0.0, le=1.0, description="风格相似度")


class HairstyleTaskResponse(BaseModel):
    """发型任务响应"""
    task_id: str = Field(..., description="任务ID")
    user_id: str = Field(..., description="用户ID")
    source_image_id: str = Field(..., description="源图片ID")
    reference_image_id: Optional[str] = Field(None, description="参考图片ID")
    task_type: str = Field(..., description="任务类型")
    input_params: Dict[str, Any] = Field(..., description="输入参数")
    status: TaskStatus = Field(..., description="任务状态")
    progress: int = Field(0, ge=0, le=100, description="处理进度")
    current_stage: Optional[str] = Field(None, description="当前处理阶段")
    error_message: Optional[str] = Field(None, description="错误信息")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间(秒)")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

    class Config:
        from_attributes = True


class HairstyleResultResponse(BaseModel):
    """发型结果响应"""
    result_id: str = Field(..., description="结果ID")
    task_id: str = Field(..., description="任务ID")
    user_id: str = Field(..., description="用户ID")
    source_image_url: str = Field(..., description="源图片URL")
    result_image_url: str = Field(..., description="结果图片URL")
    result_params: HairstyleParameters = Field(..., description="结果参数")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="质量评分")
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="用户评分")
    user_feedback: Optional[str] = Field(None, max_length=1000, description="用户反馈")
    download_count: int = Field(0, description="下载次数")
    view_count: int = Field(0, description="查看次数")
    is_public: bool = Field(False, description="是否公开")
    created_at: datetime = Field(..., description="创建时间")
    processing_time: Optional[int] = Field(None, description="处理时间(秒)")

    class Config:
        from_attributes = True


class HairstyleHistoryResponse(BaseModel):
    """发型历史响应"""
    result: HairstyleResultResponse = Field(..., description="结果信息")
    source_image: Optional[Dict[str, Any]] = Field(None, description="源图片信息")
    task: Optional[HairstyleTaskResponse] = Field(None, description="任务信息")


class HairstyleFeedbackRequest(BaseModel):
    """发型反馈请求"""
    result_id: str = Field(..., description="结果ID")
    rating: int = Field(..., ge=1, le=5, description="评分(1-5)")
    feedback: Optional[str] = Field(None, max_length=1000, description="反馈内容")


class HairstyleShareRequest(BaseModel):
    """发型分享请求"""
    result_id: str = Field(..., description="结果ID")
    share_type: str = Field("public", pattern="^(public|private)$", description="分享类型")
    expires_in: Optional[int] = Field(None, ge=3600, le=86400 * 30, description="过期时间(秒)")


class HairstyleShareResponse(BaseModel):
    """发型分享响应"""
    share_token: str = Field(..., description="分享令牌")
    share_url: str = Field(..., description="分享链接")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")


class ProgressUpdate(BaseModel):
    """进度更新消息"""
    task_id: str = Field(..., description="任务ID")
    progress: int = Field(0, ge=0, le=100, description="进度")
    stage: Optional[str] = Field(None, description="当前阶段")
    message: Optional[str] = Field(None, description="状态消息")
    estimated_time: Optional[int] = Field(None, description="预计剩余时间")


class TaskError(BaseModel):
    """任务错误消息"""
    task_id: str = Field(..., description="任务ID")
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")