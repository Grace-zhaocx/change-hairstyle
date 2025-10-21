from sqlalchemy import Column, String, Boolean, DateTime, JSON, Integer, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from app.core.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    format = Column(String(10), nullable=False)
    mime_type = Column(String(100), nullable=False)
    md5_hash = Column(String(32), nullable=False)
    face_detected = Column(Boolean, default=False)
    face_bbox = Column(JSON)  # {x, y, width, height}
    status = Column(String(20), default="uploaded", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    user = relationship("User", back_populates="images")
    source_tasks = relationship("HairstyleTask", foreign_keys="HairstyleTask.source_image_id", back_populates="source_image")
    reference_tasks = relationship("HairstyleTask", foreign_keys="HairstyleTask.reference_image_id", back_populates="reference_image")

    def __repr__(self):
        return f"<Image(id={self.id}, filename={self.original_filename}, status={self.status})>"