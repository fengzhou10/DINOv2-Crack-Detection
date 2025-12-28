import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionFusionModule(nn.Module):
    """通道注意力特征融合模块"""
    
    def __init__(self, in_channels_large=1024, in_channels_base=768, out_channels=1024, reduction_ratio=4):
        """
        初始化CAFM模块
        
        Args:
            in_channels_large: Large教师特征通道数
            in_channels_base: Base教师特征通道数
            out_channels: 输出通道数
            reduction_ratio: 通道缩减比例
        """
        super().__init__()
        
        self.in_channels = in_channels_large + in_channels_base
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        
        # 通道注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(self.in_channels, self.in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // reduction_ratio, self.in_channels),
            nn.Sigmoid()
        )
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feat_large, feat_base):
        """
        前向传播
        
        Args:
            feat_large: Large教师特征 [B, C_large, H, W]
            feat_base: Base教师特征 [B, C_base, H, W]
        
        Returns:
            融合后的特征 [B, C_out, H, W]
        """
        # 确保特征大小一致
        if feat_large.size()[2:] != feat_base.size()[2:]:
            feat_large = F.interpolate(feat_large, size=feat_base.shape[2:], 
                                      mode='bilinear', align_corners=False)
        
        # 拼接特征
        combined = torch.cat([feat_large, feat_base], dim=1)
        
        # 计算通道注意力权重
        att_weights = self.attention(combined)
        att_weights = att_weights.view(-1, self.in_channels, 1, 1)
        
        # 应用注意力权重
        attended = combined * att_weights
        
        # 融合特征
        fused = self.fusion_conv(attended)
        
        return fused
    
    def compute_flops(self, input_size):
        """计算FLOPs"""
        h, w = input_size
        
        # 注意力机制的FLOPs
        flops = 0
        
        # 全局平均池化
        flops += self.in_channels * h * w
        
        # 第一个全连接层
        flops += self.in_channels * (self.in_channels // self.reduction_ratio) * 2
        
        # 第二个全连接层
        flops += (self.in_channels // self.reduction_ratio) * self.in_channels * 2
        
        # 融合卷积的FLOPs
        # 1x1卷积
        flops += self.in_channels * self.out_channels * h * w
        
        # 3x3卷积
        flops += self.out_channels * self.out_channels * 9 * h * w
        
        return flops