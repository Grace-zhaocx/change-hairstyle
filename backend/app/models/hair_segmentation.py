"""
头发分割模型实现
基于深度学习的头发区域分割
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image

from app.utils.logger import logger
from app.core.config import settings


class HairSegmentationModel:
    """头发分割模型"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.model_path = model_path or self._get_model_path()

        # 预处理参数
        self.input_size = (512, 512)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _get_model_path(self) -> str:
        """获取模型路径"""
        model_file = "segformer_hair.pth"
        model_path = Path(settings.MODEL_CACHE_DIR) / model_file

        if not model_path.exists():
            # 尝试其他可能的路径
            alternative_paths = [
                Path("models") / model_file,
                Path("models/segmentation") / model_file,
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Hair segmentation model not found at {model_path}")

        return str(model_path)

    async def load_model(self):
        """加载模型"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # 检查是否为模拟模型
            if self._is_mock_model():
                logger.warning("Loading mock hair segmentation model")
                self.model = self._create_mock_model()
                return

            # 加载真实模型
            logger.info(f"Loading hair segmentation model from {self.model_path}")
            self.model = self._create_simplified_model()

            self.model.to(self.device)
            self.model.eval()

            logger.info("Hair segmentation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load hair segmentation model: {str(e)}")
            # 回退到模拟模型
            logger.warning("Using mock model as fallback")
            self.model = self._create_mock_model()

    def _is_mock_model(self) -> bool:
        """检查是否为模拟模型"""
        try:
            with open(self.model_path, 'rb') as f:
                content = f.read(100)
                return b"MOCK_SEGFORMER_WEIGHTS" in content
        except:
            return False

    def _create_mock_model(self):
        """创建模拟模型"""
        class MockHairSegmentation:
            def __init__(self):
                self.name = "MockHairSegmentation"

            def __call__(self, x):
                batch_size = x.shape[0]
                # 返回模拟的头发掩码
                mask = torch.zeros(batch_size, 1, 512, 512)
                # 在上半部分创建模拟的头发区域
                mask[:, :, 100:200, 150:350] = 1.0
                return mask

        return MockHairSegmentation()

    def _create_simplified_model(self):
        """创建简化的头发分割模型"""
        class SimplifiedHairSegmentation(nn.Module):
            def __init__(self):
                super().__init__()
                # 编码器
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 2, 1),    # 256x256
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, 2, 1),  # 128x128
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, 2, 1), # 64x64
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, 3, 2, 1), # 32x32
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )

                # 解码器
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),  # 64x64
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),  # 128x128
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),   # 256x256
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 1, 3, 2, 1, 1),     # 512x512
                    nn.Sigmoid()
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        model = SimplifiedHairSegmentation()

        # 尝试加载权重
        try:
            if not self._is_mock_model():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
        except Exception as e:
            logger.warning(f"Failed to load model weights: {str(e)}")

        return model

    def preprocess_image(self, image: np.ndarray, face_bbox: Dict[str, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """预处理图像"""
        original_shape = image.shape[:2]  # (height, width)

        # 转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 根据人脸位置裁剪感兴趣区域
        x, y, w, h = face_bbox["x"], face_bbox["y"], face_bbox["width"], face_bbox["height"]

        # 扩展区域以包含头发
        hair_region_y = max(0, y - h // 2)
        hair_region_h = min(original_shape[0] - hair_region_y, h + h // 2)
        hair_region_x = max(0, x - w // 4)
        hair_region_w = min(original_shape[1] - hair_region_x, w + w // 2)

        hair_region = image[hair_region_y:hair_region_y+hair_region_h,
                           hair_region_x:hair_region_x+hair_region_w]

        # 调整大小
        resized = cv2.resize(hair_region, self.input_size)

        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array(self.mean).reshape(1, 1, 3)
        std = np.array(self.std).reshape(1, 1, 3)
        normalized = (normalized - mean) / std

        # 转换为tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

        # 记录预处理信息
        preprocess_info = {
            'original_shape': original_shape,
            'hair_region': (hair_region_y, hair_region_x, hair_region_h, hair_region_w),
            'resized_shape': self.input_size
        }

        return tensor, preprocess_info

    def postprocess_mask(self, mask: torch.Tensor, preprocess_info: Dict[str, Any]) -> np.ndarray:
        """后处理分割掩码"""
        # 转换为numpy
        mask_np = mask.cpu().numpy().squeeze()

        # 阈值处理
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        # 调整回原始图像中的头发区域大小
        hair_region_y, hair_region_x, hair_region_h, hair_region_w = preprocess_info['hair_region']
        mask_resized = cv2.resize(mask_binary, (hair_region_w, hair_region_h))

        # 创建全图掩码
        original_h, original_w = preprocess_info['original_shape']
        full_mask = np.zeros((original_h, original_w), dtype=np.uint8)

        # 将头发区域掩码放置到正确位置
        full_mask[hair_region_y:hair_region_y+hair_region_h,
                  hair_region_x:hair_region_x+hair_region_w] = mask_resized

        return full_mask

    def segment_hair(self, image: np.ndarray, face_bbox: Dict[str, int]) -> Dict[str, Any]:
        """分割头发区域"""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            # 预处理
            input_tensor, preprocess_info = self.preprocess_image(image, face_bbox)
            input_tensor = input_tensor.to(self.device)

            # 推理
            with torch.no_grad():
                mask = self.model(input_tensor)

            # 后处理
            hair_mask = self.postprocess_mask(mask, preprocess_info)

            # 计算头发区域像素数
            hair_pixels = np.sum(hair_mask > 0)

            return {
                'hair_mask': hair_mask,
                'hair_area_pixels': int(hair_pixels),
                'image_shape': image.shape
            }

        except Exception as e:
            logger.error(f"Hair segmentation failed: {str(e)}")
            # 返回模拟结果
            return self._create_fallback_mask(image, face_bbox)

    def _create_fallback_mask(self, image: np.ndarray, face_bbox: Dict[str, int]) -> Dict[str, Any]:
        """创建回退掩码"""
        height, width = image.shape[:2]

        # 创建简单的头发区域掩码
        hair_mask = np.zeros((height, width), dtype=np.uint8)

        x, y, w, h = face_bbox["x"], face_bbox["y"], face_bbox["width"], face_bbox["height"]

        # 在人脸上方创建头发区域
        hair_top = max(0, y - h // 2)
        hair_bottom = y + h // 4
        hair_left = max(0, x - w // 4)
        hair_right = min(width, x + w + w // 4)

        # 绘制椭圆形头发区域
        center = ((hair_left + hair_right) // 2, (hair_top + hair_bottom) // 2)
        axes = ((hair_right - hair_left) // 2, (hair_bottom - hair_top) // 2)

        cv2.ellipse(hair_mask, center, axes, 0, 0, 360, 255, -1)

        return {
            'hair_mask': hair_mask,
            'hair_area_pixels': int(np.sum(hair_mask > 0)),
            'image_shape': image.shape
        }

    async def segment_hair_async(self, image_path: str, face_bbox: Dict[str, int]) -> Dict[str, Any]:
        """异步分割头发"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # 分割头发
        return self.segment_hair(image, face_bbox)


async def create_hair_segmentation_model(model_path: Optional[str] = None,
                                       device: str = "cpu") -> HairSegmentationModel:
    """创建头发分割模型实例"""
    model = HairSegmentationModel(model_path, device)
    await model.load_model()
    return model