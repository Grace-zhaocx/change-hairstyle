from sqlalchemy import Column, String, DateTime, JSON, Integer, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class HairstyleTask(Base):
    __tablename__ = "hairstyle_tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_image_id = Column(String(36), ForeignKey("images.id"), nullable=False)
    reference_image_id = Column(String(36), ForeignKey("images.id"), nullable=True)
    task_type = Column(String(20), nullable=False)  # 'text' or 'reference'
    input_params = Column(JSON, nullable=False)
    status = Column(String(20), default="pending", index=True)  # 'pending', 'processing', 'completed', 'failed', 'cancelled'
    progress = Column(Integer, default=0)
    current_stage = Column(String(50))
    error_message = Column(String(1000))
    processing_time = Column(Integer)  # 秒
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # 关系
    user = relationship("User", back_populates="tasks")
    source_image = relationship("Image", foreign_keys=[source_image_id], back_populates="source_tasks")
    reference_image = relationship("Image", foreign_keys=[reference_image_id], back_populates="reference_tasks")
    result = relationship("HairstyleResult", back_populates="task")

    def __repr__(self):
        return f"<HairstyleTask(id={self.id}, type={self.task_type}, status={self.status})>"