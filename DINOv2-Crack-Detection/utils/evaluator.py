import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from .metrics import calculate_all_metrics, MetricTracker


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, test_loader, device=None):
        """
        初始化评估器
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            device: 设备
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移动到设备
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, threshold=0.5):
        """
        评估模型性能
        
        Args:
            threshold: 二值化阈值
        
        Returns:
            评估指标字典
        """
        metric_tracker = MetricTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating')
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算指标
                metrics = calculate_all_metrics(outputs, masks, threshold)
                metric_tracker.update(metrics)
                
                # 更新进度条
                pbar.set_postfix({
                    'iou': metrics['iou'],
                    'crack_iou': metrics['crack_iou']
                })
        
        # 计算平均指标
        avg_metrics = metric_tracker.average()
        
        return avg_metrics
    
    def evaluate_robustness(self, noise_levels=[0.01, 0.03, 0.05]):
        """
        评估模型对噪声的鲁棒性
        
        Args:
            noise_levels: 噪声水平列表
        
        Returns:
            各噪声水平下的指标字典
        """
        robustness_results = {}
        
        for noise in noise_levels:
            print(f"\nEvaluating with noise level: {noise}")
            metric_tracker = MetricTracker()
            
            with torch.no_grad():
                for batch in self.test_loader:
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    
                    # 添加高斯噪声
                    noisy_images = images + torch.randn_like(images) * noise
                    noisy_images = torch.clamp(noisy_images, 0, 1)
                    
                    # 前向传播
                    outputs = self.model(noisy_images)
                    
                    # 计算指标
                    metrics = calculate_all_metrics(outputs, masks)
                    metric_tracker.update(metrics)
            
            # 记录结果
            robustness_results[f'noise_{noise}'] = metric_tracker.average()
        
        return robustness_results
    
    def evaluate_adversarial_robustness(self, epsilon=0.03, alpha=0.01, iterations=10):
        """
        评估模型对对抗攻击的鲁棒性
        
        Args:
            epsilon: 最大扰动幅度
            alpha: 每次迭代的步长
            iterations: 攻击迭代次数
        
        Returns:
            对抗攻击下的指标
        """
        print(f"\nEvaluating adversarial robustness with PGD (ε={epsilon})")
        metric_tracker = MetricTracker()
        
        for batch in tqdm(self.test_loader, desc='Adversarial Evaluation'):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 创建对抗样本
            adv_images = images.clone().detach().requires_grad_(True)
            
            # PGD攻击
            for _ in range(iterations):
                # 前向传播
                outputs = self.model(adv_images)
                loss = torch.nn.functional.binary_cross_entropy(outputs, masks)
                
                # 反向传播获取梯度
                loss.backward()
                
                # 更新对抗样本
                with torch.no_grad():
                    perturbation = alpha * adv_images.grad.sign()
                    adv_images.data = adv_images.data + perturbation
                    
                    # 投影到epsilon邻域
                    adv_images.data = torch.max(
                        torch.min(adv_images.data, images + epsilon),
                        images - epsilon
                    )
                    
                    # 确保在[0,1]范围内
                    adv_images.data = torch.clamp(adv_images.data, 0, 1)
                    adv_images.grad.zero_()
            
            # 在对抗样本上评估
            with torch.no_grad():
                adv_outputs = self.model(adv_images.detach())
                metrics = calculate_all_metrics(adv_outputs, masks)
                metric_tracker.update(metrics)
        
        return metric_tracker.average()
    
    def generate_predictions(self, output_dir, threshold=0.5):
        """
        生成预测结果并保存
        
        Args:
            output_dir: 输出目录
            threshold: 二值化阈值
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc='Generating Predictions')):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                names = batch['name']
                
                # 前向传播
                outputs = self.model(images)
                
                # 保存预测结果
                for i in range(len(names)):
                    # 获取预测掩码
                    pred_mask = outputs[i, 0].cpu().numpy()
                    pred_binary = (pred_mask > threshold).astype(np.uint8) * 255
                    
                    # 获取真实掩码
                    true_mask = masks[i, 0].cpu().numpy() * 255
                    
                    # 获取原始图像
                    image = images[i].cpu().numpy().transpose(1, 2, 0)
                    image = (image * np.array([0.229, 0.224, 0.225]) + 
                            np.array([0.485, 0.456, 0.406])) * 255
                    image = image.astype(np.uint8)
                    
                    # 创建可视化图像
                    vis_image = self._create_visualization(image, true_mask, pred_binary)
                    
                    # 保存结果
                    base_name = os.path.splitext(names[i])[0]
                    cv2.imwrite(os.path.join(output_dir, f'{base_name}_pred.png'), pred_binary)
                    cv2.imwrite(os.path.join(output_dir, f'{base_name}_vis.png'), vis_image)
    
    def _create_visualization(self, image, true_mask, pred_mask):
        """
        创建可视化图像
        
        Args:
            image: 原始图像 [H, W, 3]
            true_mask: 真实掩码 [H, W]
            pred_mask: 预测掩码 [H, W]
        
        Returns:
            可视化图像
        """
        # 调整大小
        if image.shape[:2] != true_mask.shape[:2]:
            image = cv2.resize(image, (true_mask.shape[1], true_mask.shape[0]))
        
        # 创建彩色掩码
        true_color = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
        true_color[true_mask > 0] = [0, 255, 0]  # 绿色表示真实裂缝
        
        pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        pred_color[pred_mask > 0] = [255, 0, 0]  # 红色表示预测裂缝
        
        # 叠加掩码
        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 0.7, true_color, 0.3, 0)
        overlay = cv2.addWeighted(overlay, 0.7, pred_color, 0.3, 0)
        
        # 创建对比图像
        comparison = np.hstack([
            image,
            cv2.cvtColor(true_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR),
            overlay
        ])
        
        return comparison
    
    def print_evaluation_summary(self, metrics):
        """打印评估摘要"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # 主要指标
        print("\nMain Metrics:")
        print("-" * 40)
        print(f"Crack IoU:      {metrics.get('crack_iou', 0):.4f}")
        print(f"IoU:            {metrics.get('iou', 0):.4f}")
        print(f"F1 Score:       {metrics.get('f1', 0):.4f}")
        print(f"Precision:      {metrics.get('precision', 0):.4f}")
        print(f"Recall:         {metrics.get('recall', 0):.4f}")
        print(f"Dice:           {metrics.get('dice', 0):.4f}")
        
        # 详细指标
        print("\nDetailed Metrics:")
        print("-" * 40)
        print(f"Pixel Accuracy: {metrics.get('pa', 0):.4f}")
        print(f"Crack PA:       {metrics.get('cpa', 0):.4f}")
        print(f"Mean PA:        {metrics.get('mpa', 0):.4f}")
        print(f"Background IoU: {metrics.get('bg_iou', 0):.4f}")
        print(f"Mean IoU:       {metrics.get('miou', 0):.4f}")
        
        print("=" * 60)