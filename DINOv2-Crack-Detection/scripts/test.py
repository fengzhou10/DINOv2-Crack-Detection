#!/usr/bin/env python3
"""
测试脚本
用于评估训练好的模型
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DeepCrackDataset, create_dataloaders
from models import (
    BaseCrackSegmenter,
    LipschitzCrackSegmenter,
    DualTeacherCrackSegmenter,
    DualTeacherLipschitzSegmenter
)
from utils.evaluator import ModelEvaluator


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
    
    print(f"Loaded model from: {checkpoint_path}")
    
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'best_metric' in checkpoint:
        print(f"Best validation metric: {checkpoint['best_metric']:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Test crack detection model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['base', 'lipschitz', 'dual_teacher', 'full'],
                       default='full', help='Type of model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/DeepCrack', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./results/test', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binary threshold')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction images')
    parser.add_argument('--robustness_test', action='store_true', help='Test model robustness')
    
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
    
    # 创建数据加载器
    print("Loading test dataset...")
    
    # 构建数据路径
    test_img_dir = os.path.join(args.data_dir, 'test_images')
    test_lab_dir = os.path.join(args.data_dir, 'test_labels')
    
    if not os.path.exists(test_img_dir):
        # 尝试使用val目录
        test_img_dir = os.path.join(args.data_dir, 'val_images')
        test_lab_dir = os.path.join(args.data_dir, 'val_labels')
    
    if not os.path.exists(test_img_dir):
        raise FileNotFoundError(f"Test images directory not found: {test_img_dir}")
    
    # 创建测试数据集
    test_dataset = DeepCrackDataset(
        img_dir=test_img_dir,
        lab_dir=test_lab_dir,
        image_size=config['data']['image_size'] if config and 'data' in config else 224,
        mode='test',
        augment=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test: {len(test_dataset)} images")
    
    # 创建评估器
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # 基本评估
    print("\n" + "=" * 60)
    print("Basic Evaluation")
    print("=" * 60)
    
    metrics = evaluator.evaluate(threshold=args.threshold)
    evaluator.print_evaluation_summary(metrics)
    
    # 保存评估结果
    result_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value:.4f}\n")
    
    # 鲁棒性测试
    if args.robustness_test:
        print("\n" + "=" * 60)
        print("Robustness Evaluation")
        print("=" * 60)
        
        # 噪声鲁棒性
        print("\nNoise Robustness:")
        noise_results = evaluator.evaluate_robustness(noise_levels=[0.01, 0.03, 0.05])
        for noise_level, metrics in noise_results.items():
            print(f"\n{noise_level}:")
            print(f"  Crack IoU: {metrics['crack_iou']:.4f}")
            print(f"  IoU: {metrics['iou']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
        
        # 对抗鲁棒性
        print("\nAdversarial Robustness:")
        adv_metrics = evaluator.evaluate_adversarial_robustness(epsilon=0.03)
        print(f"  Crack IoU: {adv_metrics['crack_iou']:.4f}")
        print(f"  IoU: {adv_metrics['iou']:.4f}")
        print(f"  F1: {adv_metrics['f1']:.4f}")
        
        # 保存鲁棒性结果
        robustness_file = os.path.join(args.output_dir, 'robustness_results.txt')
        with open(robustness_file, 'w') as f:
            f.write("Robustness Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Noise Robustness:\n")
            for noise_level, metrics in noise_results.items():
                f.write(f"\n{noise_level}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nAdversarial Robustness:\n")
            for key, value in adv_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
    
    # 保存预测结果
    if args.save_predictions:
        print("\nSaving predictions...")
        pred_dir = os.path.join(args.output_dir, 'predictions')
        evaluator.generate_predictions(pred_dir, threshold=args.threshold)
        print(f"Predictions saved to: {pred_dir}")
    
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()