from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.models.user import User
from app.schemas.user import UserCreate, UserResponse
from app.schemas.auth import TokenData
from app.core.config import settings
from app.utils.logger import logger


class AuthService:
    """认证服务"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """获取密码哈希"""
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """创建访问令牌"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
        return encoded_jwt

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """创建用户"""
        try:
            # 检查用户名是否已存在
            result = await self.db.execute(
                select(User).where(User.username == user_data.username)
            )
            if result.scalar_one_or_none():
                raise ValueError("用户名已存在")

            # 检查邮箱是否已存在
            result = await self.db.execute(
                select(User).where(User.email == user_data.email)
            )
            if result.scalar_one_or_none():
                raise ValueError("邮箱已被注册")

            # 创建用户
            hashed_password = self.get_password_hash(user_data.password)
            db_user = User(
                username=user_data.username,
                email=user_data.email,
                password_hash=hashed_password,
                status="active"
            )

            self.db.add(db_user)
            await self.db.commit()
            await self.db.refresh(db_user)

            logger.info(f"User created successfully: {db_user.email}")
            return UserResponse(
                id=str(db_user.id),
                username=db_user.username,
                email=db_user.email,
                avatar_url=db_user.avatar_url,
                status=db_user.status,
                preferences=db_user.preferences,
                created_at=db_user.created_at,
                updated_at=db_user.updated_at,
                last_login_at=db_user.last_login_at
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create user: {str(e)}")
            await self.db.rollback()
            raise ValueError("用户创建失败")

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """验证用户"""
        try:
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            user = result.scalar_one_or_none()

            if not user:
                return None

            if not self.verify_password(password, user.password_hash):
                return None

            # 更新最后登录时间
            user.last_login_at = datetime.utcnow()
            await self.db.commit()

            logger.info(f"User authenticated successfully: {email}")
            return user

        except Exception as e:
            logger.error(f"Failed to authenticate user: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        try:
            result = await self.db.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user by ID: {str(e)}")
            return None

    async def verify_token(self, token: str) -> Optional[TokenData]:
        """验证令牌"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            email: str = payload.get("sub")
            if email is None:
                return None
            token_data = TokenData(email=email)
            return token_data
        except JWTError as e:
            logger.error(f"Token verification failed: {str(e)}")
            return None

    async def update_user_password(self, user_id: str, new_password: str) -> bool:
        """更新用户密码"""
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if not user:
                return False

            user.password_hash = self.get_password_hash(new_password)
            await self.db.commit()

            logger.info(f"Password updated successfully for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update password: {str(e)}")
            await self.db.rollback()
            return False