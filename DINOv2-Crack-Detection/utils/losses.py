import torch
import torch.nn as nn
import torch.nn.functional as F


class CrackSegmentationLoss(nn.Module):
    """裂缝分割损失函数（加权BCE + Dice）"""
    
    def __init__(self, crack_weight=4.0, bce_weight=1.0, dice_weight=1.0):
        """
        初始化损失函数
        
        Args:
            crack_weight: 裂缝类别权重
            bce_weight: BCE损失权重
            dice_weight: Dice损失权重
        """
        super().__init__()
        self.crack_weight = crack_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        # BCE损失
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, pred, target):
        """
        计算损失
        
        Args:
            pred: 预测概率 [B, 1, H, W]
            target: 真实标签 [B, 1, H, W]
        
        Returns:
            总损失
        """
        # 加权BCE损失
        weights = target * (self.crack_weight - 1) + 1
        bce = self.bce_loss(pred, target)
        weighted_bce = (bce * weights).mean()
        
        # Dice损失
        smooth = 1e-5
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        # 组合损失
        total_loss = self.bce_weight * weighted_bce + self.dice_weight * dice
        
        return total_loss, weighted_bce.item(), dice.item()


class DualTeacherDistillationLoss(nn.Module):
    """双教师蒸馏损失"""
    
    def __init__(self, distill_weight=0.3, seg_weight=0.7, 
                 crack_weight=4.0, temperature=3.0,
                 large_weight=0.6, base_weight=0.4):
        """
        初始化蒸馏损失
        
        Args:
            distill_weight: 蒸馏损失权重
            seg_weight: 分割损失权重
            crack_weight: 裂缝类别权重
            temperature: 蒸馏温度
            large_weight: Large教师权重
            base_weight: Base教师权重
        """
        super().__init__()
        self.distill_weight = distill_weight
        self.seg_weight = seg_weight
        self.crack_weight = crack_weight
        self.temperature = temperature
        self.large_weight = large_weight
        self.base_weight = base_weight
        
        # 分割损失
        self.seg_loss = CrackSegmentationLoss(crack_weight=crack_weight)
        
        # 验证权重
        assert abs((large_weight + base_weight) - 1.0) < 1e-6, \
            f"Weights must sum to 1, got {large_weight} + {base_weight} = {large_weight + base_weight}"
    
    def forward(self, pred, target, teacher_large_list, teacher_base_list,
                student_large_list, student_base_list):
        """
        计算总损失
        
        Args:
            pred: 预测概率
            target: 真实标签
            teacher_large_list: Large教师特征列表
            teacher_base_list: Base教师特征列表
            student_large_list: 学生Large特征列表
            student_base_list: 学生Base特征列表
        
        Returns:
            total_loss: 总损失
            seg_loss: 分割损失
            distill_loss: 蒸馏损失
            distill_loss_large: Large教师损失
            distill_loss_base: Base教师损失
        """
        # 分割损失
        seg_loss, bce_loss, dice_loss = self.seg_loss(pred, target)
        
        # 蒸馏损失
        distill_loss, distill_loss_large, distill_loss_base = \
            self._compute_distill_loss(
                teacher_large_list, teacher_base_list,
                student_large_list, student_base_list
            )
        
        # 组合损失
        total_loss = (self.seg_weight * seg_loss + 
                     self.distill_weight * distill_loss)
        
        return (total_loss, seg_loss, distill_loss, 
                distill_loss_large, distill_loss_base, bce_loss, dice_loss)
    
    def _compute_distill_loss(self, teacher_large_list, teacher_base_list,
                             student_large_list, student_base_list):
        """计算蒸馏损失"""
        # 计算Large教师损失
        distill_loss_large = self._compute_single_teacher_loss(
            teacher_large_list, student_large_list
        )
        
        # 计算Base教师损失
        distill_loss_base = self._compute_single_teacher_loss(
            teacher_base_list, student_base_list
        )
        
        # 加权融合
        distill_loss = (self.large_weight * distill_loss_large + 
                       self.base_weight * distill_loss_base)
        
        return distill_loss, distill_loss_large, distill_loss_base
    
    def _compute_single_teacher_loss(self, teacher_features, student_features):
        """计算单教师损失"""
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


class LipschitzConstraintLoss(nn.Module):
    """Lipschitz约束损失"""
    
    def __init__(self, lambda_lip=0.05, target_L=0.8, use_gradient_penalty=True):
        """
        初始化Lipschitz损失
        
        Args:
            lambda_lip: Lipschitz损失权重
            target_L: 目标Lipschitz常数
            use_gradient_penalty: 是否使用梯度惩罚
        """
        super().__init__()
        self.lambda_lip = lambda_lip
        self.target_L = target_L
        self.use_gradient_penalty = use_gradient_penalty
    
    def forward(self, model, features, images=None):
        """
        计算Lipschitz损失
        
        Args:
            model: 模型
            features: 特征图
            images: 输入图像（用于完整模型）
        
        Returns:
            lipschitz_loss: Lipschitz损失
        """
        if not self.use_gradient_penalty:
            return torch.tensor(0.0, device=features.device)
        
        # 计算梯度惩罚
        lip_loss = self._compute_gradient_penalty(model.decoder, features)
        
        # 应用权重
        weighted_loss = self.lambda_lip * lip_loss
        
        return weighted_loss, lip_loss.item()
    
    def _compute_gradient_penalty(self, decoder, features):
        """计算梯度惩罚"""
        # 分离特征并启用梯度
        features = features.detach()
        features.requires_grad_(True)
        
        # 解码器前向传播
        decoder_output = decoder(features)
        
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
        penalty = torch.mean((grad_norms - self.target_L) ** 2)
        
        return penalty