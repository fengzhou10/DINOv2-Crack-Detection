import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

from .base import BaseCrackSegmenter
from .modules.decoder import LipschitzDecoder


class LipschitzCrackSegmenter(BaseCrackSegmenter):
    """带有Lipschitz约束的裂缝分割模型"""
    
    def __init__(self, input_size=224, backbone_type='dinov2_vitb14', lip_config=None):
        """
        初始化Lipschitz约束模型
        
        Args:
            input_size: 输入图像大小
            backbone_type: 骨干网络类型
            lip_config: Lipschitz配置字典
        """
        super().__init__(input_size, backbone_type)
        
        # Lipschitz配置
        self.lip_config = lip_config or {}
        
        # 替换解码器为Lipschitz约束解码器
        self.decoder = LipschitzDecoder(
            input_size=input_size,
            in_channels=768,
            lip_config=self.lip_config
        )
        
        # 重新初始化权重
        self._initialize_weights()
    
    def compute_lipschitz_penalty(self, features, target_L=0.8):
        """
        计算Lipschitz梯度惩罚
        
        Args:
            features: 骨干网络输出的特征
            target_L: 目标Lipschitz常数
        
        Returns:
            梯度惩罚损失
        """
        # 分离特征并启用梯度
        features = features.detach()
        features.requires_grad_(True)
        
        # 解码器前向传播
        decoder_output = self.decoder(features)
        
        # 随机方向
        dummy_grad = torch.randn_like(decoder_output)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=decoder_output,
            inputs=features,
            grad_outputs=dummy_grad,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        # 计算惩罚，添加归一化因子
        num_pixels = features.size(2) * features.size(3)
        grad_norms = torch.norm(gradients.reshape(gradients.size(0), -1), dim=1) / num_pixels
        
        # 计算相对目标差的惩罚
        penalty = torch.mean((grad_norms - target_L) ** 2)
        
        return penalty
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入图像
            return_features: 是否返回中间特征
        
        Returns:
            分割掩码或(分割掩码, 特征)
        """
        # 获取特征
        features = self.backbone.get_intermediate_layers(x, n=1, return_class_token=False)[0]
        
        # 处理特征 [B, N, D]
        batch_size, seq_len, hidden_dim = features.shape
        grid_size = int(math.sqrt(seq_len))
        
        # 重塑为2D特征图
        features = features.permute(0, 2, 1).reshape(
            batch_size, hidden_dim, grid_size, grid_size
        )
        
        # 解码器处理
        mask = self.decoder(features)
        output = torch.sigmoid(mask)
        
        if return_features:
            return output, features
        
        return output