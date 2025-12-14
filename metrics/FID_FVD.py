import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
import numpy as np
import cv2
import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import sys

# 添加pytorch-i3d路径
sys.path.append('pytorch-i3d')
from pytorch_i3d.pytorch_i3d import InceptionI3d

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FIDCalculator:
    """FID计算器"""
    
    def __init__(self, device='cuda', batch_size=50):
        self.device = device
        self.batch_size = batch_size
        self.inception_model = None
        self._load_inception_model()
    
    def _load_inception_model(self):
        """加载InceptionV3模型"""
        logger.info("加载InceptionV3模型...")
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        # 移除最后的分类层，使用pool3特征
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.inception_model.to(self.device)
        logger.info("InceptionV3模型加载完成")
    
    def _preprocess_images(self, images):
        """预处理图像"""
        if isinstance(images, list):
            # 如果是路径列表，加载图像
            processed_images = []
            for img_path in images:
                if isinstance(img_path, str):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = img_path
                
                # 调整大小到299x299
                img = cv2.resize(img, (299, 299))
                # 归一化到[0, 1]
                img = img.astype(np.float32) / 255.0
                # 转换为张量 [C, H, W]
                img = torch.from_numpy(img).permute(2, 0, 1)
                processed_images.append(img)
            
            images = torch.stack(processed_images)
        
        # 标准化（ImageNet统计量）
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        images = normalize(images)
        
        return images.to(self.device)
    
    def extract_features(self, images):
        """
        提取图像特征
        Args:
            images: 图像张量 [N, 3, 299, 299] 或图像路径列表
        Returns:
            features: [N, 2048]
        """
        images = self._preprocess_images(images)
        
        all_features = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size]
                features = self.inception_model(batch)
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def calculate_statistics(self, features):
        """
        计算特征的均值和协方差矩阵
        Args:
            features: [N, 2048]
        Returns:
            mu: 均值向量 [2048]
            sigma: 协方差矩阵 [2048, 2048]
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算FID分数
        Args:
            mu1, sigma1: 真实图像的统计量
            mu2, sigma2: 生成图像的统计量
            eps: 数值稳定性参数
        Returns:
            fid: FID分数
        """
        # 计算均值差的平方和
        diff = mu1 - mu2
        mean_diff = np.sum(diff ** 2)
        
        # 计算协方差矩阵的矩阵平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 检查数值稳定性
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 如果是复数，取实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算FID
        trace_covmean = np.trace(covmean)
        fid = mean_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean
        
        return fid
    
    def compute_fid(self, real_images, generated_images):
        """
        计算FID分数
        Args:
            real_images: 真实图像路径列表或张量
            generated_images: 生成图像路径列表或张量
        Returns:
            fid_score: FID分数
        """
        logger.info("开始计算FID...")
        
        # 提取真实图像特征
        logger.info(f"提取真实图像特征，共{len(real_images)}张图像...")
        real_features = self.extract_features(real_images)
        
        # 提取生成图像特征
        logger.info(f"提取生成图像特征，共{len(generated_images)}张图像...")
        gen_features = self.extract_features(generated_images)
        
        # 计算统计量
        logger.info("计算统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        # 计算FID
        fid_score = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        
        logger.info(f"FID计算完成: {fid_score:.2f}")
        return fid_score


class FVDCalculator:
    """FVD计算器"""
    
    def __init__(self, device='cuda', batch_size=8, num_frames=16):
        self.device = device
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.i3d_model = None
        self._load_i3d_model()
    
    def _load_i3d_model(self):
        """加载I3D模型"""
        logger.info("加载I3D模型...")
        try:
            # 尝试加载预训练模型
            model_path = 'pytorch-i3d/models/rgb_imagenet.pt'
            if os.path.exists(model_path):
                self.i3d_model = InceptionI3d(400, in_channels=3)
                self.i3d_model.load_state_dict(torch.load(model_path))
                logger.info(f"从{model_path}加载I3D模型")
            else:
                # 如果没有预训练模型，创建随机初始化的模型
                self.i3d_model = InceptionI3d(400, in_channels=3)
                logger.warning("未找到预训练I3D模型，使用随机初始化")
            
            self.i3d_model.eval()
            self.i3d_model.to(self.device)
            logger.info("I3D模型加载完成")
            
        except Exception as e:
            logger.error(f"I3D模型加载失败: {str(e)}")
            raise
    
    def _preprocess_video(self, video_path, frame_size=224):
        """
        预处理视频
        Args:
            video_path: 视频路径
            frame_size: 帧大小
        Returns:
            video_tensor: [channels, num_frames, height, width]
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            logger.warning(f"视频帧数({total_frames})少于所需帧数({self.num_frames})")
            # 重复最后一帧
            frames = []
            for i in range(self.num_frames):
                frame_idx = min(i, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (frame_size, frame_size))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
        else:
            # 均匀采样帧
            frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (frame_size, frame_size))
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"无法从视频{video_path}中提取帧")
        
        # 转换为张量 [num_frames, height, width, channels] -> [channels, num_frames, height, width]
        video_array = np.array(frames)
        video_tensor = torch.FloatTensor(video_array).permute(3, 0, 1, 2)
        
        return video_tensor
    
    def extract_video_features(self, videos):
        """
        提取视频特征
        Args:
            videos: 视频路径列表或张量列表
        Returns:
            features: [N, 1024]
        """
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(videos), self.batch_size):
                batch_videos = videos[i:i+self.batch_size]
                batch_tensors = []
                
                for video in batch_videos:
                    if isinstance(video, str):
                        video_tensor = self._preprocess_video(video)
                    else:
                        video_tensor = video
                    batch_tensors.append(video_tensor)
                
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    # 使用I3D提取特征
                    features = self.i3d_model.extract_features(batch_tensor)
                    # 全局平均池化
                    features = F.adaptive_avg_pool3d(features, (1, 1, 1))
                    features = features.squeeze(-1).squeeze(-1).squeeze(-1)
                    all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def calculate_statistics(self, features):
        """计算特征的均值和协方差矩阵"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fvd(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算FVD分数（使用与FID相同的公式）"""
        # 计算均值差的平方和
        diff = mu1 - mu2
        mean_diff = np.sum(diff ** 2)
        
        # 计算协方差矩阵的矩阵平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 检查数值稳定性
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 如果是复数，取实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算FVD
        trace_covmean = np.trace(covmean)
        fvd = mean_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean
        
        return fvd
    
    def compute_fvd(self, real_videos, generated_videos):
        """
        计算FVD分数
        Args:
            real_videos: 真实视频路径列表
            generated_videos: 生成视频路径列表
        Returns:
            fvd_score: FVD分数
        """
        logger.info("开始计算FVD...")
        
        # 提取真实视频特征
        logger.info(f"提取真实视频特征，共{len(real_videos)}个视频...")
        real_features = self.extract_video_features(real_videos)
        
        # 提取生成视频特征
        logger.info(f"提取生成视频特征，共{len(generated_videos)}个视频...")
        gen_features = self.extract_video_features(generated_videos)
        
        # 计算统计量
        logger.info("计算统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        # 计算FVD
        fvd_score = self.calculate_fvd(mu_real, sigma_real, mu_gen, sigma_gen)
        
        logger.info(f"FVD计算完成: {fvd_score:.2f}")
        return fvd_score


def compute_fid(real_images, generated_images, device='cuda', batch_size=50):
    """
    计算FID分数的便捷函数
    Args:
        real_images: 真实图像路径列表或张量
        generated_images: 生成图像路径列表或张量
        device: 计算设备
        batch_size: 批处理大小
    Returns:
        fid_score: FID分数
    """
    calculator = FIDCalculator(device=device, batch_size=batch_size)
    return calculator.compute_fid(real_images, generated_images)


def compute_fvd(real_videos, generated_videos, device='cuda', batch_size=8, num_frames=16):
    """
    计算FVD分数的便捷函数
    Args:
        real_videos: 真实视频路径列表
        generated_videos: 生成视频路径列表
        device: 计算设备
        batch_size: 批处理大小
        num_frames: 每个视频采样的帧数
    Returns:
        fvd_score: FVD分数
    """
    calculator = FVDCalculator(device=device, batch_size=batch_size, num_frames=num_frames)
    return calculator.compute_fvd(real_videos, generated_videos)


if __name__ == "__main__":
    # 测试代码
    print("FID/FVD计算器测试")
    
    # 创建一些测试数据
    test_images = [np.random.rand(299, 299, 3) for _ in range(10)]
    test_videos = ["test_video1.mp4", "test_video2.mp4"]  # 需要实际的视频文件
    
    try:
        # 测试FID计算
        print("测试FID计算...")
        fid_calculator = FIDCalculator(device='cpu', batch_size=5)
        fid_score = fid_calculator.compute_fid(test_images, test_images)
        print(f"FID分数: {fid_score:.2f}")
        
        # 测试FVD计算（如果有视频文件）
        if all(os.path.exists(v) for v in test_videos):
            print("测试FVD计算...")
            fvd_calculator = FVDCalculator(device='cpu', batch_size=2, num_frames=8)
            fvd_score = fvd_calculator.compute_fvd(test_videos, test_videos)
            print(f"FVD分数: {fvd_score:.2f}")
        else:
            print("跳过FVD测试（缺少视频文件）")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

