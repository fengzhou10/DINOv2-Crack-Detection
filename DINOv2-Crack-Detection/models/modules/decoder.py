import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class BaseDecoder(nn.Module):
    """基础解码器模块"""
    
    def __init__(self, input_size=224, in_channels=768, out_channels=1):
        """
        初始化基础解码器
        
        Args:
            input_size: 输入特征大小
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()
        self.input_size = input_size
        
        # 解码器结构
        self.decoder = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 第二层
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 第三层
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 第四层
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 输出层
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False),
        )
        
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
    
    def forward(self, x):
        """前向传播"""
        return self.decoder(x)


class LipschitzDecoder(nn.Module):
    """带有Lipschitz约束的解码器"""
    
    def __init__(self, input_size=224, in_channels=1024, out_channels=1, lip_config=None):
        """
        初始化Lipschitz解码器
        
        Args:
            input_size: 输入特征大小
            in_channels: 输入通道数
            out_channels: 输出通道数
            lip_config: Lipschitz配置字典
        """
        super().__init__()
        
        self.input_size = input_size
        self.lip_config = lip_config or {}
        
        # 解码器层配置
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(self._make_layer(in_channels, 512, use_sn=True))
        self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # 第二层
        self.layers.append(self._make_layer(512, 256, use_sn=True))
        self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # 第三层
        self.layers.append(self._make_layer(256, 128, use_sn=True))
        self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
        # 第四层
        self.layers.append(self._make_layer(128, 64, use_sn=True))
        
        # 输出层（不使用谱归一化）
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_upsample = nn.Upsample(
            size=(input_size, input_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, use_sn=False):
        """创建解码器层"""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 应用谱归一化
        if use_sn and self.lip_config.get('use_spectral_norm', False):
            conv = spectral_norm(conv)
        
        layer = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        return layer
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_orig'):  # 谱归一化卷积
                    nn.init.kaiming_normal_(m.weight_orig, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_conv(x)
        x = self.final_upsample(x)
        
        return x
    
    def compute_lipschitz_constant(self):
        """计算解码器的Lipschitz常数上界"""
        # 对于谱归一化层，每层的Lipschitz常数被约束为1
        lip_constant = 1.0
        
        # 对于非谱归一化层，使用权重范数估计
        for name, param in self.named_parameters():
            if 'weight' in name and 'output_conv' not in name:
                # 估计卷积层的Lipschitz常数
                if len(param.shape) == 4:  # 卷积权重
                    # 使用权重的谱范数作为估计
                    with torch.no_grad():
                        u, s, v = torch.svd(param.view(param.shape[0], -1))
                        lip_constant *= s[0].item()
        
        return lip_constant