import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseCrackSegmenter
from .modules.cafm import ChannelAttentionFusionModule
from .modules.distillation import FeatureDistillationModule


class DualTeacherCrackSegmenter(BaseCrackSegmenter):
    """双教师知识蒸馏裂缝分割模型"""
    
    def __init__(self, input_size=224, backbone_type='dinov2_vitb14', distill_config=None):
        """
        初始化双教师模型
        
        Args:
            input_size: 输入图像大小
            backbone_type: 骨干网络类型
            distill_config: 蒸馏配置字典
        """
        super().__init__(input_size, backbone_type)
        
        # 蒸馏配置
        self.distill_config = distill_config or {}
        
        # 加载教师模型
        self.teacher_large = self._load_teacher('dinov2_vitl14')
        self.teacher_base = self._load_teacher('dinov2_vitb14')
        
        # 冻结教师模型
        for teacher in [self.teacher_large, self.teacher_base]:
            for param in teacher.parameters():
                param.requires_grad = False
        
        # 特征适配器
        self.feature_adapters_large = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        
        self.feature_adapters_base = nn.ModuleList([
            nn.Identity() for _ in range(3)
        ])
        
        # 通道注意力特征融合模块
        self.feature_fusion = ChannelAttentionFusionModule(
            in_channels_large=1024,
            in_channels_base=768,
            out_channels=1024
        )
        
        # 蒸馏模块
        self.distillation = FeatureDistillationModule(
            temperature=self.distill_config.get('temperature', 3.0),
            large_weight=self.distill_config.get('large_weight', 0.6),
            base_weight=self.distill_config.get('base_weight', 0.4)
        )
        
        # 更新解码器以接受融合特征
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False),
        )
        
        # 特征中心化参数
        self.center_large = None
        self.center_base = None
        self.center_momentum = 0.9
        
        # 蒸馏层配置
        self.distill_layers_base = self.distill_config.get('distill_layers_base', [3, 7, 11])
        self.distill_layers_large = self.distill_config.get('distill_layers_large', [7, 15, 23])
        
        self._initialize_weights()
    
    def _load_teacher(self, teacher_type):
        """加载教师模型"""
        try:
            return torch.hub.load('facebookresearch/dinov2', teacher_type)
        except:
            import sys
            sys.path.append('..')
            return torch.hub.load(".", teacher_type, source="local")
    
    def get_features(self, model, x, layers):
        """获取指定层的特征图"""
        features = []
        x = model.prepare_tokens_with_masks(x)
        
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in layers:
                # 获取特征图并重塑为空间格式
                batch_size, num_tokens, hidden_dim = x.shape
                grid_size = int(math.sqrt(num_tokens - 1))
                
                # 提取图像token（排除class token）
                image_tokens = x[:, 1:]
                
                # 重塑为 (batch_size, hidden_dim, grid_size, grid_size)
                image_features = image_tokens.permute(0, 2, 1).view(
                    batch_size, hidden_dim, grid_size, grid_size
                )
                features.append(image_features)
        
        return features
    
    def update_center(self, teacher_features_large, teacher_features_base):
        """更新特征中心"""
        # Large教师中心更新
        if self.center_large is None:
            self.center_large = []
            for feat in teacher_features_large:
                self.center_large.append(torch.mean(feat.detach(), dim=0, keepdim=True))
        else:
            for i in range(len(teacher_features_large)):
                self.center_large[i] = (self.center_large[i] * self.center_momentum + 
                                       torch.mean(teacher_features_large[i].detach(), dim=0, keepdim=True) * 
                                       (1 - self.center_momentum))
        
        # Base教师中心更新
        if self.center_base is None:
            self.center_base = []
            for feat in teacher_features_base:
                self.center_base.append(torch.mean(feat.detach(), dim=0, keepdim=True))
        else:
            for i in range(len(teacher_features_base)):
                self.center_base[i] = (self.center_base[i] * self.center_momentum + 
                                      torch.mean(teacher_features_base[i].detach(), dim=0, keepdim=True) * 
                                      (1 - self.center_momentum))
    
    def center_features(self, features, teacher_type):
        """中心化特征"""
        if teacher_type == "large":
            center = self.center_large
        else:
            center = self.center_base
        
        if center is None:
            return features
        
        centered = []
        for i, feat in enumerate(features):
            centered.append(feat - center[i])
        return centered
    
    def forward(self, x, return_features=False):
        """
        前向传播
        
        Args:
            x: 输入图像
            return_features: 是否返回中间特征
        
        Returns:
            分割掩码或(分割掩码, 教师特征, 学生特征)
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
                   adapted_features_large, adapted_features_base)
        
        return output
    
    def compute_distillation_loss(self, teacher_large, teacher_base, student_large, student_base):
        """计算蒸馏损失"""
        return self.distillation(
            teacher_large, teacher_base, student_large, student_base
        )
    
    def get_parameters(self, lr_backbone=1e-5, lr_decoder=1e-4):
        """获取不同学习率的参数组"""
        backbone_params = []
        decoder_params = []
        adapter_params = []
        fusion_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name and 'teacher' not in name:
                backbone_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            elif 'feature_adapters' in name:
                adapter_params.append(param)
            elif 'feature_fusion' in name:
                fusion_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': decoder_params, 'lr': lr_decoder},
            {'params': adapter_params, 'lr': lr_decoder},
            {'params': fusion_params, 'lr': lr_decoder},
        ]