import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationModule(nn.Module):
    """特征蒸馏模块"""
    
    def __init__(self, temperature=3.0, large_weight=0.6, base_weight=0.4):
        """
        初始化蒸馏模块
        
        Args:
            temperature: 温度参数
            large_weight: Large教师权重
            base_weight: Base教师权重
        """
        super().__init__()
        
        self.temperature = temperature
        self.large_weight = large_weight
        self.base_weight = base_weight
        
        # 验证权重和是否为1
        total_weight = large_weight + base_weight
        if abs(total_weight - 1.0) > 1e-6:
            # 归一化权重
            self.large_weight = large_weight / total_weight
            self.base_weight = base_weight / total_weight
    
    def forward(self, teacher_large_list, teacher_base_list, 
                student_large_list, student_base_list):
        """
        计算蒸馏损失
        
        Args:
            teacher_large_list: Large教师特征列表
            teacher_base_list: Base教师特征列表
            student_large_list: 学生Large特征列表
            student_base_list: 学生Base特征列表
        
        Returns:
            total_loss: 总蒸馏损失
            large_loss: Large教师损失
            base_loss: Base教师损失
        """
        # 计算Large教师的蒸馏损失
        large_loss = self._compute_layer_distill_loss(
            teacher_large_list, student_large_list
        )
        
        # 计算Base教师的蒸馏损失
        base_loss = self._compute_layer_distill_loss(
            teacher_base_list, student_base_list
        )
        
        # 加权融合
        total_loss = self.large_weight * large_loss + self.base_weight * base_loss
        
        return total_loss, large_loss, base_loss
    
    def _compute_layer_distill_loss(self, teacher_features, student_features):
        """
        计算单教师的多层蒸馏损失
        
        Args:
            teacher_features: 教师特征列表
            student_features: 学生特征列表
        
        Returns:
            蒸馏损失
        """
        distill_loss = 0.0
        num_layers = len(teacher_features)
        
        for i in range(num_layers):
            teacher_feat = teacher_features[i]
            student_feat = student_features[i]
            
            # 归一化特征
            teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
            student_norm = F.normalize(student_feat, p=2, dim=1)
            
            # 计算软目标分布
            teacher_probs = F.softmax(teacher_norm / self.temperature, dim=1)
            student_log_probs = F.log_softmax(student_norm / self.temperature, dim=1)
            
            # KL散度损失
            layer_loss = F.kl_div(
                student_log_probs, teacher_probs,
                reduction='batchmean', log_target=False
            ) * (self.temperature ** 2)
            
            # 加权平均（深层权重更高）
            layer_weight = (i + 1) / num_layers
            distill_loss += layer_weight * layer_loss
        
        return distill_loss
    
    def compute_feature_similarity(self, teacher_features, student_features):
        """
        计算特征相似度
        
        Args:
            teacher_features: 教师特征
            student_features: 学生特征
        
        Returns:
            相似度矩阵
        """
        batch_size = teacher_features.shape[0]
        
        # 展平特征
        teacher_flat = teacher_features.view(batch_size, -1)
        student_flat = student_features.view(batch_size, -1)
        
        # 归一化
        teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
        student_norm = F.normalize(student_flat, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(teacher_norm, student_norm.transpose(0, 1))
        
        return similarity