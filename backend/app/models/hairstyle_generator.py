"""
发型生成模型实现
基于Stable Diffusion的AI发型生成
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image, ImageDraw
import logging

# Hugging Face diffusers
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from app.utils.logger import logger
from app.core.config import settings


class HairstyleGenerator:
    """基于Stable Diffusion的发型生成器"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path or "runwayml/stable-diffusion-inpainting"
        self.pipeline = None
        self.loaded = False

        # 生成参数
        self.default_guidance_scale = 7.5
        self.default_num_inference_steps = 20
        self.default_strength = 0.8

        # 发型特定的负面提示词
        self.negative_prompts = [
            "ugly, bad quality, distorted, deformed",
            "blurry, low resolution, pixelated",
            "bad anatomy, disfigured, malformed",
            "extra limbs, missing limbs, fused fingers",
            "watermark, signature, text, writing",
            "cropped, out of frame, cut off"
        ]

    async def load_model(self, progress_callback=None):
        """加载发型生成模型 - 优先使用真实AI模型"""
        try:
            if self.loaded:
                return

            logger.info(f"Loading hairstyle generation model: {self.model_path}")

            # 检查设备可用性
            if self.device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")

            # 尝试加载真实的Stable Diffusion模型
            try:
                logger.info("Attempting to load Stable Diffusion model (this may take a few minutes for first download)...")
                if progress_callback:
                    progress_callback("Downloading Stable Diffusion model...", 10)

                # 定义下载进度回调
                def download_progress_callback(progress):
                    if progress_callback:
                        # Hugging Face下载进度通常是0-100
                        progress_percent = min(10 + int(progress * 0.7), 80)  # 10-80%用于下载
                        progress_callback(f"Downloading model files... {int(progress)}%", progress_percent)

                # 加载模型
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    safety_checker=None,  # 禁用安全检查器以避免限制
                    requires_safety_checker=False,
                    # local_files_only=False,  # 允许下载
                    # resume_download=True    # 支持断点续传
                )

                if progress_callback:
                    progress_callback("Loading model into memory...", 85)

                # 优化调度器
                if progress_callback:
                    progress_callback("Optimizing model scheduler...", 90)

                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )

                # 移动到指定设备
                if progress_callback:
                    progress_callback("Moving model to device...", 95)

                self.pipeline = self.pipeline.to(self.device)

                # 内存优化
                if self.device.type == "cuda":
                    self.pipeline.enable_model_cpu_offload()
                    self.pipeline.enable_xformers_memory_efficient_attention()

                self.loaded = True
                if progress_callback:
                    progress_callback("Model loaded successfully!", 100)
                logger.info("Stable Diffusion model loaded successfully")

            except Exception as e:
                logger.warning(f"Failed to load Stable Diffusion model: {str(e)}")
                logger.info("Falling back to generator implementation...")
                self._create_fallback_generator()

        except Exception as e:
            logger.error(f"Failed to load hairstyle model: {str(e)}")
            # 强制使用回退实现
            self._create_fallback_generator()

    def _create_fallback_generator(self):
        """创建回退生成器"""
        logger.warning("Using fallback hairstyle generator")
        self.pipeline = self._FallbackGenerator()
        self.loaded = True

    def build_hairstyle_prompt(self, description: Dict[str, Any]) -> str:
        """构建发型特定的提示词"""
        length_map = {
            "short": "short hair, bob cut, pixie cut",
            "medium": "medium length hair, shoulder-length hair",
            "long": "long hair, flowing hair, wavy long hair"
        }

        style_map = {
            "straight": "straight hair, smooth hair, sleek hair, silky texture, no waves, no curls",
            "wavy": "wavy hair, soft waves, beach waves, gentle texture",
            "curly": "curly hair, bouncy curls, ringlets, defined curls",
            "braided": "braided hair, french braid, cornrows, woven style",
            "buzzed": "buzz cut, short shaved hair, very short"
        }

        color_map = {
            "natural": "natural hair color, black hair",
            "brown": "brown hair, brunette, chocolate brown",
            "blonde": "blonde hair, golden blonde, platinum blonde",
            "black": "jet black hair, raven black hair",
            "red": "red hair, ginger hair, auburn hair"
        }

        prompt_parts = []

        # 基础描述
        prompt_parts.append("professional photography, portrait, beautiful hairstyle")

        # 长度
        if length := description.get("length"):
            prompt_parts.append(length_map.get(length, "hair"))

        # 风格
        if style := description.get("style"):
            prompt_parts.append(style_map.get(style, "styled hair"))

        # 颜色
        if color := description.get("color"):
            prompt_parts.append(color_map.get(color, "hair"))

        # 自定义描述
        if custom_desc := description.get("custom_description"):
            prompt_parts.append(custom_desc)

        # 质量描述
        prompt_parts.extend([
            "high quality, detailed, sharp focus",
            "professional lighting, studio lighting",
            "natural skin texture, realistic",
            "8k resolution, photorealistic"
        ])

        return ", ".join(prompt_parts)

    def build_negative_prompt(self) -> str:
        """构建负面提示词"""
        return ", ".join(self.negative_prompts)

    def preprocess_images(self, image_path: str, mask_array: np.ndarray) -> Tuple[Image.Image, Image.Image]:
        """预处理图像和掩码"""
        # 加载原始图像
        original_image = Image.open(image_path).convert("RGB")

        # 确保掩码是正确的格式
        if mask_array.dtype != np.uint8:
            mask_array = (mask_array * 255).astype(np.uint8)

        # 调整掩码大小以匹配图像
        mask_pil = Image.fromarray(mask_array, mode="L")
        if mask_pil.size != original_image.size:
            mask_pil = mask_pil.resize(original_image.size, Image.LANCZOS)

        # 扩展掩码区域以确保覆盖整个头发
        mask_array_expanded = np.array(mask_pil)

        # 形态学操作扩展掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_array_expanded = cv2.morphologyEx(mask_array_expanded, cv2.MORPH_CLOSE, kernel)
        mask_array_expanded = cv2.morphologyEx(mask_array_expanded, cv2.MORPH_DILATE, kernel)

        # 模拟边缘
        mask_array_expanded = cv2.GaussianBlur(mask_array_expanded, (5, 5), 2)

        expanded_mask = Image.fromarray(mask_array_expanded, mode="L")

        return original_image, expanded_mask

    async def generate_hairstyle(
        self,
        image_path: str,
        hair_mask: np.ndarray,
        description: Dict[str, Any],
        parameters: Dict[str, float],
        num_samples: int = 1
    ) -> List[str]:
        """生成新发型"""
        try:
            if not self.loaded:
                await self.load_model()

            # 预处理图像
            original_image, mask_image = self.preprocess_images(image_path, hair_mask)

            # 构建提示词
            prompt = self.build_hairstyle_prompt(description)
            negative_prompt = self.build_negative_prompt()

            # 获取生成参数
            guidance_scale = parameters.get("guidance_scale", self.default_guidance_scale)
            num_inference_steps = int(parameters.get("num_inference_steps", self.default_num_inference_steps))
            strength = parameters.get("strength", self.default_strength)

            logger.info(f"Generating hairstyle with prompt: {prompt[:100]}...")

            # 生成图像
            if hasattr(self.pipeline, '__call__') and not isinstance(self.pipeline, self._FallbackGenerator):
                # 真实的Stable Diffusion推理
                with torch.no_grad():
                    result = self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=original_image,
                        mask_image=mask_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        num_images_per_prompt=num_samples,
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    )

                generated_images = result.images
            else:
                # 回退生成器
                generated_images = self.pipeline(original_image, mask_image, prompt)

            # 保存结果
            result_paths = []
            for i, img in enumerate(generated_images):
                result_path = self._save_generated_image(img, i)
                result_paths.append(result_path)

            logger.info(f"Hairstyle generation completed: {len(result_paths)} images")
            return result_paths

        except Exception as e:
            logger.error(f"Hairstyle generation failed: {str(e)}")
            raise

    def _save_generated_image(self, image: Image.Image, index: int = 0) -> str:
        """保存生成的图像"""
        from datetime import datetime
        import uuid

        result_dir = Path("data/results")
        result_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"hairstyle_{timestamp}_{unique_id}_{index}.jpg"
        result_path = result_dir / filename

        # 保存高质量图像
        image.save(result_path, "JPEG", quality=95, optimize=True)

        return str(result_path)

    class _FallbackGenerator:
        """回退生成器（当真实模型加载失败时使用）"""

        def __init__(self):
            logger.warning("Initialized fallback hairstyle generator")

        def __call__(self, image: Image.Image, mask: Image.Image, prompt: str) -> List[Image.Image]:
            """简单的图像处理模拟发型生成"""
            try:
                # 转换为numpy数组
                img_array = np.array(image)
                mask_array = np.array(mask)

                # 确保掩码是二值的
                mask_binary = (mask_array > 128).astype(np.uint8)

                # 在头发区域应用一些图像处理效果
                result = img_array.copy()

                # 获取头发区域
                hair_region = mask_binary == 1

                if np.any(hair_region):
                    # 简单的颜色调整模拟新发型
                    # 亮度调整
                    result[hair_region] = np.clip(result[hair_region] * 1.1, 0, 255).astype(np.uint8)

                    # 轻微的色调调整
                    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                    hsv[hair_region, 0] = (hsv[hair_region, 0] + 10) % 180  # 色调偏移
                    hsv[hair_region, 1] = np.clip(hsv[hair_region, 1] * 1.2, 0, 255)  # 饱和度增加
                    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                # 转换回PIL图像
                result_image = Image.fromarray(result)

                return [result_image]

            except Exception as e:
                logger.error(f"Fallback generator failed: {str(e)}")
                # 如果连回退都失败，返回原图
                return [image]


async def create_hairstyle_generator(model_path: Optional[str] = None,
                                  device: str = "cpu",
                                  progress_callback=None) -> HairstyleGenerator:
    """创建发型生成器实例"""
    generator = HairstyleGenerator(model_path, device)
    await generator.load_model(progress_callback)
    return generator