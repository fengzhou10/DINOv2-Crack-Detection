import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.utils import make_grid


def visualize_batch(images, masks, predictions=None, num_images=4):
    """
    可视化批次数据
    
    Args:
        images: 图像张量 [B, C, H, W]
        masks: 真实掩码张量 [B, 1, H, W]
        predictions: 预测掩码张量 [B, 1, H, W] (可选)
        num_images: 要可视化的图像数量
    
    Returns:
        matplotlib图像
    """
    # 限制图像数量
    num_images = min(num_images, images.shape[0])
    
    # 转换为numpy数组
    images_np = images[:num_images].cpu().numpy()
    masks_np = masks[:num_images].cpu().numpy()
    
    if predictions is not None:
        predictions_np = predictions[:num_images].cpu().numpy()
    
    # 反标准化图像
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images_np = images_np * std + mean
    images_np = np.clip(images_np, 0, 1)
    images_np = images_np.transpose(0, 2, 3, 1)
    
    # 创建子图
    fig, axes = plt.subplots(num_images, 3 if predictions is not None else 2, 
                            figsize=(12, 4*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # 显示原始图像
        axes[i, 0].imshow(images_np[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # 显示真实掩码
        axes[i, 1].imshow(masks_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # 显示预测掩码
        if predictions is not None:
            axes[i, 2].imshow(predictions_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_attention(features, num_features=16):
    """
    可视化注意力特征图
    
    Args:
        features: 特征张量 [B, C, H, W]
        num_features: 要可视化的特征数量
    
    Returns:
        matplotlib图像
    """
    # 选择批次中的第一个样本
    features = features[0].cpu().detach()
    
    # 限制特征数量
    num_features = min(num_features, features.shape[0])
    features = features[:num_features]
    
    # 归一化每个特征图
    normalized_features = []
    for i in range(num_features):
        feat = features[i].numpy()
        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
        normalized_features.append(feat)
    
    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_features):
        axes[i].imshow(normalized_features[i], cmap='viridis')
        axes[i].set_title(f'Feature {i+1}')
        axes[i].axis('off')
    
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_predictions_comparison(image, true_mask, pred_mask, threshold=0.5):
    """
    可视化预测结果对比
    
    Args:
        image: 原始图像 [H, W, 3]
        true_mask: 真实掩码 [H, W]
        pred_mask: 预测概率图 [H, W]
        threshold: 二值化阈值
    
    Returns:
        matplotlib图像
    """
    # 二值化预测
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    
    # 创建彩色掩码
    true_color = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    true_color[true_mask > 0] = [0, 255, 0]  # 绿色
    
    pred_color = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    pred_color[pred_binary > 0] = [255, 0, 0]  # 红色
    
    # 计算重叠区域
    overlap = np.logical_and(true_mask > 0, pred_binary > 0)
    overlap_color = np.zeros((*overlap.shape, 3), dtype=np.uint8)
    overlap_color[overlap] = [255, 255, 0]  # 黄色
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 真实掩码
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # 预测概率图
    im = axes[0, 2].imshow(pred_mask, cmap='hot')
    axes[0, 2].set_title('Prediction Probability')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 二值化预测
    axes[1, 0].imshow(pred_binary, cmap='gray')
    axes[1, 0].set_title(f'Binary Prediction (threshold={threshold})')
    axes[1, 0].axis('off')
    
    # 真实和预测对比
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(true_color, alpha=0.5)
    axes[1, 1].imshow(pred_color, alpha=0.5)
    axes[1, 1].set_title('Overlay: Green=True, Red=Pred')
    axes[1, 1].axis('off')
    
    # 重叠区域
    axes[1, 2].imshow(image)
    axes[1, 2].imshow(overlap_color, alpha=0.7)
    axes[1, 2].set_title('Overlap Area (Yellow)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    return fig


def save_visualization(image, true_mask, pred_mask, save_path, threshold=0.5):
    """
    保存可视化结果到文件
    
    Args:
        image: 原始图像
        true_mask: 真实掩码
        pred_mask: 预测掩码
        save_path: 保存路径
        threshold: 二值化阈值
    """
    # 确保是numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * np.array([0.229, 0.224, 0.225]) + 
                np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
    
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy().squeeze()
    
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy().squeeze()
    
    # 二值化预测
    pred_binary = (pred_mask > threshold).astype(np.uint8) * 255
    
    # 创建彩色掩码
    true_color = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    true_color[true_mask > 0] = [0, 255, 0]  # 绿色
    
    pred_color = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    pred_color[pred_binary > 0] = [255, 0, 0]  # 红色
    
    # 创建对比图像
    comparison = np.hstack([
        image,
        cv2.cvtColor((true_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)
    ])
    
    # 保存图像
    cv2.imwrite(save_path, comparison)
    
    return comparison


def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, save_dir=None):
    """
    绘制训练历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标字典
        val_metrics: 验证指标字典
        save_dir: 保存目录
    """
    epochs = len(train_losses)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='Train')
    axes[0, 0].plot(val_losses, label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU曲线
    if 'iou' in train_metrics and 'iou' in val_metrics:
        axes[0, 1].plot(train_metrics['iou'], label='Train')
        axes[0, 1].plot(val_metrics['iou'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('IoU Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Crack IoU曲线
    if 'crack_iou' in train_metrics and 'crack_iou' in val_metrics:
        axes[0, 2].plot(train_metrics['crack_iou'], label='Train')
        axes[0, 2].plot(val_metrics['crack_iou'], label='Val')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Crack IoU')
        axes[0, 2].set_title('Crack IoU Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
    
    # F1曲线
    if 'f1' in train_metrics and 'f1' in val_metrics:
        axes[1, 0].plot(train_metrics['f1'], label='Train')
        axes[1, 0].plot(val_metrics['f1'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Precision-Recall曲线
    if 'precision' in train_metrics and 'recall' in train_metrics:
        axes[1, 1].plot(train_metrics['precision'], label='Train Precision')
        axes[1, 1].plot(train_metrics['recall'], label='Train Recall')
        if 'precision' in val_metrics and 'recall' in val_metrics:
            axes[1, 1].plot(val_metrics['precision'], label='Val Precision')
            axes[1, 1].plot(val_metrics['recall'], label='Val Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Dice曲线
    if 'dice' in train_metrics and 'dice' in val_metrics:
        axes[1, 2].plot(train_metrics['dice'], label='Train')
        axes[1, 2].plot(val_metrics['dice'], label='Val')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Dice Coefficient')
        axes[1, 2].set_title('Dice Coefficient')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    return fig