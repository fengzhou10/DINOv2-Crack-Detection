import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BaseCrackSegmenter(nn.Module):
    """基础裂缝分割模型（DINOv2-base + 简单解码器）"""
    
    def __init__(self, input_size=224, backbone_type='dinov2_vitb14'):
        """
        初始化基础模型
        
        Args:
            input_size: 输入图像大小
            backbone_type: 骨干网络类型
        """
        super().__init__()
        self.input_size = input_size
        
        # 加载DINOv2-base模型
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_type)
        except:
            # 如果无法从网络加载，尝试从本地加载
            import sys
            sys.path.append('..')
            self.backbone = torch.hub.load(".", backbone_type, source="local")
        
        # 冻结骨干网络的部分参数（可选）
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        # 解码器配置
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
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
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            分割掩码 [B, 1, H, W]
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
        
        # Sigmoid激活
        return torch.sigmoid(mask)
    
    def get_intermediate_features(self, x, layer_indices=None):
        """
        获取中间层特征
        
        Args:
            x: 输入图像
            layer_indices: 要获取的层索引
        
        Returns:
            中间层特征列表
        """
        if layer_indices is None:
            layer_indices = [3, 7, 11]  # DINOv2-base的典型层
        
        features = []
        x = self.backbone.prepare_tokens_with_masks(x)
        
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in layer_indices:
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
    
    def get_parameters(self, lr_backbone=1e-5, lr_decoder=1e-4):
        """获取不同学习率的参数组"""
        backbone_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': decoder_params, 'lr': lr_decoder},
        ]