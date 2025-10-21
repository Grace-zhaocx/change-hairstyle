from sqlalchemy import Column, String, DateTime, JSON, Integer, ForeignKey, Boolean, Numeric
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class HairstyleResult(Base):
    __tablename__ = "hairstyle_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("hairstyle_tasks.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_image_id = Column(String(36), ForeignKey("images.id"), nullable=False)
    result_image_id = Column(String(36), ForeignKey("images.id"), nullable=False)
    result_params = Column(JSON)
    quality_score = Column(Numeric(3, 2))  # 0.00-1.00
    user_rating = Column(Integer)  # 1-5
    user_feedback = Column(String(1000))
    share_token = Column(String(100), unique=True, index=True)
    share_expires_at = Column(DateTime(timezone=True))
    download_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    is_public = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    user = relationship("User", back_populates="results")
    task = relationship("HairstyleTask", back_populates="result")
    source_image = relationship("Image", foreign_keys=[source_image_id])
    result_image = relationship("Image", foreign_keys=[result_image_id])

    def __repr__(self):
        return f"<HairstyleResult(id={self.id}, user_id={self.user_id}, rating={self.user_rating})>"