#!/usr/bin/env python3
"""
预下载模型脚本
在 Docker 构建时预下载所需的模型文件，使用国内镜像源加速下载
"""

import os
import sys
import torch
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 设置环境变量（使用镜像源）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/app/Hugging_Face')
os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', '/app/weights')

def download_huggingface_model():
    """下载 HuggingFace 模型：chatterbox-tts"""
    try:
        logger.info("=" * 60)
        logger.info("下载 HuggingFace 模型: chatterbox-tts")
        logger.info("=" * 60)
        
        # 尝试导入 chatterbox
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError:
            logger.warning("chatterbox-tts 未安装，跳过下载")
            logger.warning("提示: 需要先安装 chatterbox-tts: pip install chatterbox-tts")
            return False
        
        # 下载模型（使用 CPU 设备，因为构建时可能没有 GPU）
        device = 'cpu'
        logger.info(f"使用设备: {device}")
        logger.info(f"HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
        logger.info(f"HuggingFace 缓存目录: {os.environ['HF_HOME']}")
        
        model = ChatterboxMultilingualTTS.from_pretrained(device)
        logger.info("✅ chatterbox-tts 模型下载成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ chatterbox-tts 模型下载失败: {e}")
        return False

def download_torch_hub_models():
    """下载 PyTorch Hub 模型"""
    try:
        logger.info("=" * 60)
        logger.info("下载 PyTorch Hub 模型")
        logger.info("=" * 60)
        
        torch_home = Path(os.environ['TORCH_HOME'])
        hub_dir = torch_home / 'hub' / 'checkpoints'
        hub_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Torch Hub 缓存目录: {hub_dir}")
        
        models_to_download = [
            {
                'name': 'resnet18',
                'repo': 'pytorch/vision',
                'model': 'resnet18',
                'file': 'resnet18-5c106cde.pth'
            },
            {
                'name': 'alexnet',
                'repo': 'pytorch/vision',
                'model': 'alexnet',
                'file': 'alexnet-owt-7be5be79.pth'
            }
        ]
        
        success_count = 0
        for model_info in models_to_download:
            try:
                logger.info(f"\n下载 {model_info['name']}...")
                model = torch.hub.load(
                    model_info['repo'],
                    model_info['model'],
                    pretrained=True,
                    progress=True
                )
                logger.info(f"✅ {model_info['name']} 下载成功")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ {model_info['name']} 下载失败: {e}")
        
        logger.info(f"\n成功下载 {success_count}/{len(models_to_download)} 个模型")
        return success_count == len(models_to_download)
        
    except Exception as e:
        logger.error(f"❌ PyTorch Hub 模型下载失败: {e}")
        return False

def download_3dfan4_model():
    """下载 3DFAN4 模型（face_alignment 使用）"""
    try:
        logger.info("=" * 60)
        logger.info("下载 3DFAN4 模型")
        logger.info("=" * 60)
        
        torch_home = Path(os.environ['TORCH_HOME'])
        hub_dir = torch_home / 'hub' / 'checkpoints'
        hub_dir.mkdir(parents=True, exist_ok=True)
        
        # 3DFAN4 模型文件
        model_file = hub_dir / '3DFAN4-4a694010b9.zip'
        model_url = 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip'
        
        if model_file.exists():
            logger.info(f"✅ 3DFAN4 模型已存在: {model_file}")
            return True
        
        logger.info(f"下载地址: {model_url}")
        logger.info(f"保存位置: {model_file}")
        
        # 使用 requests 下载（如果可用）
        try:
            import requests
            from tqdm import tqdm
            
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(model_file, 'wb') as f, tqdm(
                desc="下载 3DFAN4",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✅ 3DFAN4 模型下载成功: {model_file}")
            return True
            
        except ImportError:
            logger.warning("requests 或 tqdm 未安装，尝试使用 urllib...")
            import urllib.request
            
            urllib.request.urlretrieve(model_url, model_file)
            logger.info(f"✅ 3DFAN4 模型下载成功: {model_file}")
            return True
            
    except Exception as e:
        logger.error(f"❌ 3DFAN4 模型下载失败: {e}")
        logger.warning("提示: 可以稍后手动下载或运行时自动下载")
        return False

def download_depth_model():
    """下载深度估计模型 (depth-6c4283c0e0)"""
    try:
        logger.info("=" * 60)
        logger.info("下载深度估计模型")
        logger.info("=" * 60)
        
        torch_home = Path(os.environ['TORCH_HOME'])
        hub_dir = torch_home / 'hub' / 'checkpoints'
        hub_dir.mkdir(parents=True, exist_ok=True)
        
        # depth-6c4283c0e0 可能是 face_alignment 或其他库使用的深度模型
        # 尝试通过 torch.hub 下载常见的深度估计模型
        
        # 方法1: 尝试下载 MiDaS 模型（常见的深度估计模型）
        try:
            logger.info("尝试下载 MiDaS 深度估计模型...")
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
            logger.info("✅ MiDaS 模型下载成功")
            return True
        except Exception as e:
            logger.warning(f"MiDaS 模型下载失败: {e}")
        
        # 方法2: 尝试直接下载 depth-6c4283c0e0 文件（如果是已知的模型文件）
        # 这个哈希值可能是某个特定模型的标识
        model_file = hub_dir / 'depth-6c4283c0e0.pth'
        if model_file.exists():
            logger.info(f"✅ 深度模型已存在: {model_file}")
            return True
        
        # 如果无法确定具体模型，记录警告但继续
        logger.warning("无法确定 depth-6c4283c0e0 的具体模型")
        logger.warning("如果运行时需要，将自动下载")
        return False
            
    except Exception as e:
        logger.error(f"❌ 深度估计模型下载失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始预下载模型文件")
    logger.info("=" * 60)
    logger.info(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    logger.info("")
    
    results = {}
    
    # 1. 下载 HuggingFace 模型
    results['chatterbox'] = download_huggingface_model()
    
    # 2. 下载 PyTorch Hub 模型
    results['torch_hub'] = download_torch_hub_models()
    
    # 3. 下载 3DFAN4 模型
    results['3dfan4'] = download_3dfan4_model()
    
    # 4. 下载深度估计模型
    results['depth'] = download_depth_model()
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("下载总结")
    logger.info("=" * 60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"{name:20s}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    logger.info(f"\n总计: {success_count}/{total_count} 成功")
    
    # 如果所有关键模型都下载成功，返回 0，否则返回 1
    # chatterbox 和 torch_hub 是关键模型
    if results.get('chatterbox', False) and results.get('torch_hub', False):
        logger.info("\n✅ 关键模型下载完成")
        return 0
    else:
        logger.warning("\n⚠️  部分关键模型下载失败，但构建将继续")
        return 0  # 即使失败也继续构建，允许运行时下载

if __name__ == '__main__':
    sys.exit(main())

