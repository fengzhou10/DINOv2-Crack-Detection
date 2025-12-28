#!/usr/bin/env python3
"""
训练脚本
支持训练不同配置的模型：基础模型、Lipschitz模型、双教师模型、完整模型
"""

import os
import sys
import argparse
import yaml
import torch
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
from utils.trainer import BaseTrainer
from utils.losses import (
    CrackSegmentationLoss,
    DualTeacherDistillationLoss,
    LipschitzConstraintLoss
)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type, config, device):
    """根据类型创建模型"""
    model_config = config['model']
    
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
    
    return model.to(device)


def create_trainer(model_type, model, train_loader, val_loader, config, device):
    """创建训练器"""
    if model_type in ['base', 'lipschitz']:
        return BaseTrainer(model, train_loader, val_loader, config, device)
    
    elif model_type in ['dual_teacher', 'full']:
        # 对于双教师模型，使用自定义训练器
        from utils.dual_teacher_trainer import DualTeacherTrainer
        return DualTeacherTrainer(model, train_loader, val_loader, config, device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Train crack detection model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['base', 'lipschitz', 'dual_teacher', 'full'],
                       default='full', help='Type of model to train')
    parser.add_argument('--data_dir', type=str, default='./data/DeepCrack', help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据加载器
    print("Loading datasets...")
    data_config = config['data']
    
    # 更新数据路径
    if args.data_dir:
        data_config['train_img_dir'] = os.path.join(args.data_dir, 'train_images')
        data_config['train_lab_dir'] = os.path.join(args.data_dir, 'train_labels')
        data_config['val_img_dir'] = os.path.join(args.data_dir, 'val_images')
        data_config['val_lab_dir'] = os.path.join(args.data_dir, 'val_labels')
        data_config['test_img_dir'] = os.path.join(args.data_dir, 'test_images')
        data_config['test_lab_dir'] = os.path.join(args.data_dir, 'test_labels')
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_config,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )
    
    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Val: {len(val_loader.dataset)} images")
    print(f"Test: {len(test_loader.dataset)} images")
    
    # 创建模型
    print(f"\nCreating {args.model_type} model...")
    model = create_model(args.model_type, config, device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = create_trainer(args.model_type, model, train_loader, val_loader, 
                           config['training'], device)
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("\nStarting training...")
    best_metric = trainer.train()
    
    print(f"\nTraining completed. Best Crack IoU: {best_metric:.4f}")
    
    # 测试最佳模型
    print("\nTesting best model...")
    from utils.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # 加载最佳模型
    checkpoint_path = os.path.join(trainer.result_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # 评估
    metrics = evaluator.evaluate()
    evaluator.print_evaluation_summary(metrics)
    
    # 保存评估结果
    result_file = os.path.join(trainer.result_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write("Test Results\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value:.4f}\n")
    
    print(f"\nResults saved to: {trainer.result_dir}")


if __name__ == '__main__':
    main()