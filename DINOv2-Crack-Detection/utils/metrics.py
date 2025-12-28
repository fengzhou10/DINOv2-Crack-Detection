import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score


def calculate_iou(pred, target, threshold=0.5):
    """
    计算IoU（交并比）
    
    Args:
        pred: 预测概率图 [B, 1, H, W] 或 [B, H, W]
        target: 真实标签 [B, 1, H, W] 或 [B, H, W]
        threshold: 二值化阈值
    
    Returns:
        IoU值
    """
    # 确保是4D张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算交集和并集
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # 避免除零
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    """
    计算Dice系数
    
    Args:
        pred: 预测概率图
        target: 真实标签
        threshold: 二值化阈值
    
    Returns:
        Dice系数
    """
    # 确保是4D张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算Dice
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + 1e-6) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    
    return dice.item()


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    计算像素准确率
    
    Args:
        pred: 预测概率图
        target: 真实标签
        threshold: 二值化阈值
    
    Returns:
        像素准确率
    """
    # 确保是4D张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算准确率
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    
    accuracy = correct / total
    
    return accuracy.item()


def calculate_precision_recall_f1(pred, target, threshold=0.5):
    """
    计算精确率、召回率和F1分数
    
    Args:
        pred: 预测概率图
        target: 真实标签
        threshold: 二值化阈值
    
    Returns:
        precision, recall, f1
    """
    # 转换为numpy数组
    if torch.is_tensor(pred):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    
    if torch.is_tensor(target):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target
    
    # 展平
    pred_flat = (pred_np > threshold).astype(np.uint8).flatten()
    target_flat = (target_np > threshold).astype(np.uint8).flatten()
    
    # 计算指标
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    return precision, recall, f1


def calculate_crack_iou(pred, target, threshold=0.5):
    """
    计算裂缝区域的IoU（Crack IoU）
    
    Args:
        pred: 预测概率图
        target: 真实标签
        threshold: 二值化阈值
    
    Returns:
        裂缝IoU
    """
    return calculate_iou(pred, target, threshold)


def calculate_all_metrics(pred, target, threshold=0.5):
    """
    计算所有评估指标
    
    Args:
        pred: 预测概率图 [B, 1, H, W]
        target: 真实标签 [B, 1, H, W]
        threshold: 二值化阈值
    
    Returns:
        包含所有指标的字典
    """
    # 确保是4D张量
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    # 二值化
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # 计算基本指标
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    # IoU
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice
    dice = (2. * intersection + 1e-6) / (pred_binary.sum() + target_binary.sum() + 1e-6)
    
    # 像素准确率
    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    pa = correct / total
    
    # 类别像素准确率
    true_pos = (pred_binary * target_binary).sum()
    false_pos = (pred_binary * (1 - target_binary)).sum()
    false_neg = ((1 - pred_binary) * target_binary).sum()
    true_neg = ((1 - pred_binary) * (1 - target_binary)).sum()
    
    # 背景和裂缝的准确率
    bg_acc = true_neg / (true_neg + false_pos + 1e-6)
    crack_acc = true_pos / (true_pos + false_neg + 1e-6)
    
    # 平均像素准确率
    mpa = (bg_acc + crack_acc) / 2
    
    # 类别IoU
    bg_iou = true_neg / (true_neg + false_pos + false_neg + 1e-6)
    crack_iou = intersection / (union + 1e-6)
    miou = (bg_iou + crack_iou) / 2
    
    # 精确率、召回率、F1
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 转换为标量
    metrics = {
        'iou': iou.item(),
        'dice': dice.item(),
        'pa': pa.item(),
        'cpa': crack_acc.item(),  # 裂缝像素准确率
        'mpa': mpa.item(),        # 平均像素准确率
        'bg_iou': bg_iou.item(),
        'crack_iou': crack_iou.item(),
        'miou': miou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
    
    return metrics


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
    
    def update(self, metrics_dict):
        """更新指标"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def average(self):
        """计算平均指标"""
        avg_metrics = {}
        for key, values in self.metrics.items():
            avg_metrics[key] = np.mean(values)
        return avg_metrics
    
    def summary(self):
        """打印指标摘要"""
        avg_metrics = self.average()
        print("\nMetrics Summary:")
        print("-" * 50)
        for key, value in avg_metrics.items():
            print(f"{key:15s}: {value:.4f}")
        print("-" * 50)
        
        return avg_metrics