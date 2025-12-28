import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .dual_teacher import DualTeacherCrackSegmenter
from .modules.decoder import LipschitzDecoder


class DualTeacherLipschitzSegmenter(DualTeacherCrackSegmenter):
    """完整模型：双教师蒸馏 + Lipschitz约束"""
    
    def __init__(self, input_size=224, backbone_type='dinov2_vitb14', 
                 distill_config=None, lip_config=None):
        """
        初始化完整模型
        
        Args:
            input_size: 输入图像大小
            backbone_type: 骨干网络类型
            distill_config: 蒸馏配置字典
            lip_config: Lipschitz配置字典
        """
        super().__init__(input_size, backbone_type, distill_config)
        
        # Lipschitz配置
        self.lip_config = lip_config or {}
        
        # 替换解码器为Lipschitz约束解码器
        self.decoder = LipschitzDecoder(
            input_size=input_size,
            in_channels=1024,  # 融合后的特征维度
            lip_config=self.lip_config
        )
        
        # 重新初始化权重
        self._initialize_weights()
    
    def compute_lipschitz_penalty(self, fused_feature, target_L=0.8):
        """
        计算Lipschitz梯度惩罚
        
        Args:
            fused_feature: 融合后的特征
            target_L: 目标Lipschitz常数
        
        Returns:
            梯度惩罚损失
        """
        # 分离特征并启用梯度
        features = fused_feature.detach()
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
            分割掩码或(分割掩码, 教师特征, 学生特征, 融合特征)
        """
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_features_large = self.get_features(self.teacher_large, x, self.distill_layers_large)
            teacher_features_base = self.get_features(self.teacher_base, x, self.distill_layers_base)
        
        # 学生模型特征
        student_features = self.get_features(self.backbone, x, self.distill_layers_base)
        
        # 特征适配
        adapted_features_large = []
        adapted_features_base = []
        for i in range(len(self.distill_layers_base)):
            adapted_large = self.feature_adapters_large[i](student_features[i])
            adapted_base = self.feature_adapters_base[i](student_features[i])
            adapted_features_large.append(adapted_large)
            adapted_features_base.append(adapted_base)
        
        # 更新特征中心
        self.update_center(teacher_features_large, teacher_features_base)
        
        # 中心化教师特征
        teacher_features_large = self.center_features(teacher_features_large, "large")
        teacher_features_base = self.center_features(teacher_features_base, "base")
        
        # 特征融合
        fused_feature = self.feature_fusion(adapted_features_large[-1], adapted_features_base[-1])
        
        # 解码
        mask = self.decoder(fused_feature)
        output = torch.sigmoid(mask)
        
        if return_features:
            return (output, teacher_features_large, teacher_features_base, 
                   adapted_features_large, adapted_features_base, fused_feature)
        
        return output