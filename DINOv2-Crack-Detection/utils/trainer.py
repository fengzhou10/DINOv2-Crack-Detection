import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime

from .losses import CrackSegmentationLoss, DualTeacherDistillationLoss, LipschitzConstraintLoss
from .metrics import calculate_all_metrics, MetricTracker


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, model, train_loader, val_loader, config, device=None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
            device: 设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(
            config.get('result_dir', 'results'),
            f"{config.get('model_name', 'model')}_{timestamp}"
        )
        os.makedirs(self.result_dir, exist_ok=True)
        
        # TensorBoard写入器
        self.writer = SummaryWriter(log_dir=os.path.join(self.result_dir, 'tensorboard'))
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = CrackSegmentationLoss(
            crack_weight=config.get('crack_weight', 4.0)
        )
        
        # 最佳模型指标
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        print(f"Results will be saved to: {self.result_dir}")
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr_backbone = self.config.get('lr_backbone', 1e-5)
        lr_decoder = self.config.get('lr_decoder', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        # 获取参数组
        if hasattr(self.model, 'get_parameters'):
            params = self.model.get_parameters(lr_backbone, lr_decoder)
        else:
            params = [
                {'params': self.model.backbone.parameters(), 'lr': lr_backbone},
                {'params': self.model.decoder.parameters(), 'lr': lr_decoder},
            ]
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(params, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(params, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 50)
        
        if scheduler_type.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=epochs//3, gamma=0.1
            )
        elif scheduler_type.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        
        return scheduler
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        metric_tracker = MetricTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 数据转移到设备
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss, bce_loss, dice_loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 计算指标
            with torch.no_grad():
                metrics = calculate_all_metrics(outputs, masks)
                metric_tracker.update(metrics)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'iou': metrics['iou']
            })
            
            # TensorBoard记录
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar('train/bce_loss', bce_loss, global_step)
            self.writer.add_scalar('train/dice_loss', dice_loss, global_step)
            self.writer.add_scalar('train/iou', metrics['iou'], global_step)
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = metric_tracker.average()
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss, bce_loss, dice_loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 计算指标
                metrics = calculate_all_metrics(outputs, masks)
                metric_tracker.update(metrics)
                
                # 更新进度条
                pbar.set_postfix({
                    'iou': metrics['iou'],
                    'crack_iou': metrics['crack_iou']
                })
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = metric_tracker.average()
        
        # TensorBoard记录
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/iou', avg_metrics['iou'], epoch)
        self.writer.add_scalar('val/crack_iou', avg_metrics['crack_iou'], epoch)
        self.writer.add_scalar('val/f1', avg_metrics['f1'], epoch)
        
        return avg_loss, avg_metrics
    
    def train(self):
        """完整训练流程"""
        print("\nStarting training...")
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate(epoch)
            
            # 学习率调整
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['crack_iou'])
            else:
                self.scheduler.step()
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"Train Crack IoU: {train_metrics['crack_iou']:.4f} | Val Crack IoU: {val_metrics['crack_iou']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['crack_iou'] > self.best_metric:
                self.best_metric = val_metrics['crack_iou']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"New best model saved with Crack IoU: {self.best_metric:.4f}")
            
            # 定期保存检查点
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best Crack IoU: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        return self.best_metric
    
    def save_checkpoint(self, filename):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.result_dir, filename))
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint['best_metric']
        self.best_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best Crack IoU: {checkpoint['best_metric']:.4f}")
        
        return checkpoint['epoch']