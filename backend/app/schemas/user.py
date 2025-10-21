from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime


class UserBase(BaseModel):
    """用户基础信息"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")


class UserCreate(UserBase):
    """用户创建请求"""
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    confirm_password: str = Field(..., description="确认密码")

    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('密码确认不匹配')
        return v

    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('用户名只能包含字母、数字、下划线和连字符')
        return v


class UserLogin(BaseModel):
    """用户登录请求"""
    email: EmailStr = Field(..., description="邮箱地址")
    password: str = Field(..., description="密码")


class UserUpdate(BaseModel):
    """用户更新请求"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="用户名")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="用户偏好设置")


class UserResponse(UserBase):
    """用户响应"""
    id: str = Field(..., description="用户ID")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    status: str = Field(..., description="用户状态")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="用户偏好设置")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    last_login_at: Optional[datetime] = Field(None, description="最后登录时间")

    class Config:
        from_attributes = True


class UserProfile(UserResponse):
    """用户详细信息"""
    # 添加更多用户统计信息
    total_uploads: int = Field(0, description="总上传次数")
    total_processes: int = Field(0, description="总处理次数")
    storage_used: int = Field(0, description="已使用存储空间(字节)")
    api_quota_remaining: int = Field(0, description="剩余API配额")