#!/usr/bin/env python3
"""
可视化脚本
用于可视化模型预测结果
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DeepCrackDataset
from models import (
    BaseCrackSegmenter,
    LipschitzCrackSegmenter,
    DualTeacherCrackSegmenter,
    DualTeacherLipschitzSegmenter
)
from utils.visualization import (
    visualize_predictions_comparison,
    save_visualization,
    visualize_attention
)


def load_model(checkpoint_path, model_type, config, device):
    """加载模型"""
    # 加载配置
    if config is None:
        config = {'model': {}}
    
    model_config = config.get('model', {})
    
    # 根据类型创建模型
    if model_type == 'base':
        model = BaseCrackSegmenter(
            input_size=model_config.get('input_size', 224),
            backbone_type=model_config.get('backbone_type', 'dinov2_vitb14')
        )
    elif model_type == 'lipschitz':
        model = LipschitzCrackSegmenter(
            input_size=model_config.get('input_size', 224),
            backbone_type=model_config.get('backbone_type', 'dinov2_vitb14'),
            lip_config=model_config.get('lip_config', {})
        )
    elif model_type == 'dual_teacher':
        model = DualTeacherCrackSegmenter(
            input_size=model_config.get('input_size', 224),
            backbone_type=model_config.get('backbone_type', 'dinov2_vitb14'),
            distill_config=model_config.get('distill_config', {})
        )
    elif model_type == 'full':
        model = DualTeacherLipschitzSegmenter(
            input_size=model_config.get('input_size', 224),
            backbone_type=model_config.get('backbone_type', 'dinov2_vitb14'),
            distill_config=model_config.get('distill_config', {}),
            lip_config=model_config.get('lip_config', {})
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 尝试直接加载
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def visualize_single_image(model, image_path, label_path, device, output_path=None, 
                          model_type='full', threshold=0.5):
    """可视化单张图像的预测结果"""
    # 加载图像
    from PIL import Image
    import torchvision.transforms as transforms
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 加载图像和标签
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(label_path).convert('L') if label_path else None
    
    # 应用变换
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    if mask is not None:
        mask_tensor = mask_transform(mask).unsqueeze(0).to(device)
        mask_tensor = (mask_tensor > 0.5).float()
    else:
        mask_tensor = None
    
    # 预测
    with torch.no_grad():
        if model_type in ['dual_teacher', 'full']:
            # 对于双教师模型，可能需要额外的参数
            output = model(image_tensor, return_features=False)
        else:
            output = model(image_tensor)
        
        # 获取特征图（如果可能）
        if hasattr(model, 'get_intermediate_features'):
            features = model.get_intermediate_features(image_tensor)
        else:
            features = None
    
    # 转换为numpy数组
    image_np = image_tensor[0].cpu().numpy()
    pred_np = output[0, 0].cpu().numpy()
    
    if mask_tensor is not None:
        mask_np = mask_tensor[0, 0].cpu().numpy()
    else:
        mask_np = None
    
    # 反标准化图像
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    image_np = image_np.transpose(1, 2, 0)
    
    # 创建可视化
    if mask_np is not None:
        fig = visualize_predictions_comparison(image_np, mask_np, pred_np, threshold)
    else:
        # 只显示预测结果
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(pred_np, cmap='hot')
        axes[1].set_title('Prediction Probability')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 可视化特征图（如果可用）
    if features is not None:
        fig = visualize_attention(features[0] if len(features) > 0 else features)
        
        if output_path:
            feat_path = output_path.replace('.png', '_features.png')
            plt.savefig(feat_path, dpi=300, bbox_inches='tight')
            print(f"Feature visualization saved to: {feat_path}")
        else:
            plt.show()
        
        plt.close()
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['base', 'lipschitz', 'dual_teacher', 'full'],
                       default='full', help='Type of model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--label_path', type=str, help='Path to ground truth label (optional)')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results/visualizations', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print(f"\nLoading {args.model_type} model...")
    model = load_model(args.checkpoint, args.model_type, config, device)
    
    # 可视化单张图像
    if args.image_path:
        print(f"\nVisualizing single image: {args.image_path}")
        
        output_path = os.path.join(args.output_dir, 'single_prediction.png')
        _ = visualize_single_image(
            model, args.image_path, args.label_path, device,
            output_path, args.model_type, args.threshold
        )
    
    # 可视化数据集中的多张图像
    elif args.data_dir:
        print(f"\nVisualizing images from dataset: {args.data_dir}")
        
        # 构建数据路径
        img_dir = os.path.join(args.data_dir, 'test_images')
        lab_dir = os.path.join(args.data_dir, 'test_labels')
        
        if not os.path.exists(img_dir):
            img_dir = os.path.join(args.data_dir, 'val_images')
            lab_dir = os.path.join(args.data_dir, 'val_labels')
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Images directory not found: {img_dir}")
        
        # 创建数据集
        dataset = DeepCrackDataset(
            img_dir=img_dir,
            lab_dir=lab_dir,
            image_size=config['data']['image_size'] if config and 'data' in config else 224,
            mode='test',
            augment=False
        )
        
        # 可视化多张图像
        num_images = min(args.num_images, len(dataset))
        indices = list(range(num_images))
        
        for idx in indices:
            # 获取数据
            sample = dataset[idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            mask_tensor = sample['mask'].unsqueeze(0).to(device)
            image_name = sample['name']
            
            # 预测
            with torch.no_grad():
                if args.model_type in ['dual_teacher', 'full']:
                    output = model(image_tensor, return_features=False)
                else:
                    output = model(image_tensor)
            
            # 保存可视化结果
            base_name = os.path.splitext(image_name)[0]
            output_path = os.path.join(args.output_dir, f'{base_name}_prediction.png')
            
            # 转换为numpy数组
            image_np = image_tensor[0].cpu().numpy()
            pred_np = output[0, 0].cpu().numpy()
            mask_np = mask_tensor[0, 0].cpu().numpy()
            
            # 反标准化图像
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
            image_np = image_np.transpose(1, 2, 0)
            
            # 创建并保存可视化
            fig = visualize_predictions_comparison(image_np, mask_np, pred_np, args.threshold)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization for {image_name}")
    
    else:
        print("\nPlease provide either --image_path or --data_dir")
        sys.exit(1)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()