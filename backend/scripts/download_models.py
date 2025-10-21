#!/usr/bin/env python3
"""
下载AI模型文件脚本
"""

import os
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from app.utils.logger import logger
from app.core.config import settings


def download_file(url: str, destination: Path) -> bool:
    """下载文件"""
    try:
        logger.info(f"Downloading {url} to {destination}")

        # 创建目录
        destination.parent.mkdir(parents=True, exist_ok=True)

        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Successfully downloaded {destination.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False


def download_huggingface_model(repo_id: str, filename: str, destination: Path) -> bool:
    """从Hugging Face下载模型"""
    try:
        logger.info(f"Downloading {repo_id}/{filename} from Hugging Face")

        # 创建目录
        destination.parent.mkdir(parents=True, exist_ok=True)

        # 下载模型
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=settings.MODEL_CACHE_DIR,
            local_dir=destination.parent,
            local_dir_use_symlinks=False
        )

        # 如果文件路径和目标不一致，则复制
        if str(file_path) != str(destination):
            import shutil
            shutil.copy2(file_path, destination)

        logger.info(f"Successfully downloaded {destination.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {repo_id}/{filename}: {str(e)}")
        return False


def main():
    """主函数"""
    logger.info("Starting model downloads...")

    model_dir = Path(settings.MODEL_CACHE_DIR)
    model_dir.mkdir(exist_ok=True)

    downloads = []

    # 人脸检测模型
    downloads.append({
        "type": "huggingface",
        "repo_id": "RizwanMunawar/retina-face",
        "filename": "retinaface_resnet50.pth",
        "destination": model_dir / "retinaface_resnet50.pth"
    })

    # 头发分割模型
    downloads.append({
        "type": "huggingface",
        "repo_id": "hustvl/segformer",
        "filename": "segformer_b2_coco.pth",
        "destination": model_dir / "segformer_b2_coco.pth"
    })

    # Stable Diffusion模型
    downloads.append({
        "type": "huggingface",
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "filename": "v1-5-pruned-emaonly.safetensors",
        "destination": model_dir / "stable-diffusion-v1-5.safetensors"
    })

    # ControlNet模型
    downloads.append({
        "type": "huggingface",
        "repo_id": "lllyasviel/sd-controlnet-face",
        "filename": "control_v11p_sd15_openpose.pth",
        "destination": model_dir / "controlnet_face.pth"
    })

    success_count = 0
    total_count = len(downloads)

    for download in downloads:
        if download["type"] == "huggingface":
            success = download_huggingface_model(
                download["repo_id"],
                download["filename"],
                download["destination"]
            )
        else:
            success = download_file(
                download["url"],
                download["destination"]
            )

        if success:
            success_count += 1

    logger.info(f"Download completed: {success_count}/{total_count} models downloaded")

    if success_count == total_count:
        logger.info("All models downloaded successfully!")
        return True
    else:
        logger.warning(f"Some models failed to download. {success_count}/{total_count} successful.")
        return False


if __name__ == "__main__":
    main()