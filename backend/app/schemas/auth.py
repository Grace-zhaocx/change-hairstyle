from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    """用户创建请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$', description="邮箱地址")
    password: str = Field(..., min_length=6, max_length=100, description="密码")


class UserResponse(BaseModel):
    """用户响应"""
    id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    avatar_url: str = Field(None, description="头像URL")
    status: str = Field(..., description="用户状态")
    created_at: str = Field(..., description="创建时间")


class Token(BaseModel):
    """JWT令牌响应"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(None, description="过期时间(秒)")


class TokenData(BaseModel):
    """令牌数据"""
    email: str = Field(..., description="用户邮箱")


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class PasswordResetRequest(BaseModel):
    """密码重置请求"""
    email: str = Field(..., description="邮箱地址")


class PasswordResetConfirm(BaseModel):
    """确认密码重置"""
    token: str = Field(..., description="重置令牌")
    new_password: str = Field(..., min_length=6, max_length=100, description="新密码")
    confirm_password: str = Field(..., description="确认密码")