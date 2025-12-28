import os
import sys
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


def prepare_deepcrack_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    准备DeepCrack数据集
    
    Args:
        input_dir: 原始数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    # 创建输出目录结构
    output_dir = Path(output_dir)
    dirs = ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']
    for d in dirs:
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_dir = Path(input_dir) / 'images'
    label_dir = Path(input_dir) / 'labels'
    
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    print(f"Found {len(image_files)} images")
    
    # 分割数据集
    train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=val_ratio/(1-train_ratio), random_state=42)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # 复制文件
    def copy_files(files, split_name):
        for img_path in files:
            # 构建标签路径
            label_name = img_path.stem + '.png'
            label_path = label_dir / label_name
            
            # 复制图像
            dst_img = output_dir / f'{split_name}_images' / img_path.name
            shutil.copy(img_path, dst_img)
            
            # 复制标签（如果存在）
            if label_path.exists():
                dst_label = output_dir / f'{split_name}_labels' / label_name
                shutil.copy(label_path, dst_label)
            else:
                print(f"Warning: Label file {label_path} not found")
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"\nDataset prepared at: {output_dir}")
    print(f"Directory structure:")
    print(f"  {output_dir}/")
    for d in dirs:
        num_files = len(list((output_dir / d).glob('*')))
        print(f"    ├── {d}/ ({num_files} files)")
    
    return output_dir


def check_dataset_consistency(data_dir):
    """检查数据集一致性"""
    data_dir = Path(data_dir)
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        img_dir = data_dir / f'{split}_images'
        label_dir = data_dir / f'{split}_labels'
        
        if not img_dir.exists():
            issues.append(f"Missing directory: {img_dir}")
            continue
        
        if not label_dir.exists():
            issues.append(f"Missing directory: {label_dir}")
            continue
        
        # 检查文件数量是否匹配
        images = sorted([f.stem for f in img_dir.glob('*')])
        labels = sorted([f.stem for f in label_dir.glob('*')])
        
        if len(images) != len(labels):
            issues.append(f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels")
        
        # 检查文件名是否匹配
        for img, lbl in zip(images, labels):
            if img != lbl:
                issues.append(f"Filename mismatch in {split}: {img}.jpg vs {lbl}.png")
    
    if issues:
        print("Dataset consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Dataset is consistent!")
        return True


def analyze_dataset(data_dir):
    """分析数据集统计信息"""
    data_dir = Path(data_dir)
    
    for split in ['train', 'val', 'test']:
        img_dir = data_dir / f'{split}_images'
        label_dir = data_dir / f'{split}_labels'
        
        if not img_dir.exists():
            continue
        
        images = list(img_dir.glob('*'))
        labels = list(label_dir.glob('*'))
        
        if images:
            # 分析标签中的裂缝像素比例
            crack_pixels = []
            total_pixels = []
            
            for label_file in labels[:10]:  # 只分析前10个文件
                try:
                    mask = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        crack_pixels.append(np.sum(mask > 0))
                        total_pixels.append(mask.size)
                except:
                    pass
            
            if crack_pixels and total_pixels:
                avg_crack_ratio = np.mean([c/t for c, t in zip(crack_pixels, total_pixels)])
                print(f"{split}: {len(images)} images")
                print(f"  Average crack pixel ratio: {avg_crack_ratio:.4f}")
                print(f"  Crack pixels: {np.mean(crack_pixels):.0f} ± {np.std(crack_pixels):.0f}")
                print(f"  Total pixels: {np.mean(total_pixels):.0f}")


def main():
    parser = argparse.ArgumentParser(description='Prepare DeepCrack dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, default='./data/DeepCrack', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--check', action='store_true', help='Check dataset consistency')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset statistics')
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset_consistency(args.output_dir)
    elif args.analyze:
        analyze_dataset(args.output_dir)
    else:
        prepare_deepcrack_dataset(
            args.input_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )


if __name__ == '__main__':
    main()