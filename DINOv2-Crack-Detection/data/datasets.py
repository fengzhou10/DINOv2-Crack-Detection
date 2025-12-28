import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepCrackDataset(Dataset):
    """DeepCrack数据集类"""
    
    def __init__(self, 
                 img_dir, 
                 lab_dir, 
                 image_size=224,
                 mode='train',
                 augment=False,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录路径
            lab_dir: 标签目录路径
            image_size: 图像大小
            mode: 模式 ('train', 'val', 'test')
            augment: 是否进行数据增强
            normalize_mean: 标准化均值
            normalize_std: 标准化标准差
        """
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.image_size = image_size
        self.mode = mode
        self.augment = augment
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # 获取文件列表
        self.img_files = sorted([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # 定义数据增强
        if augment and mode == 'train':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2(),
            ])
        
        # 标签转换
        self.mask_transform = A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ])
        
        print(f"Loaded {len(self.img_files)} images for {mode} set")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 构建图像路径
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 构建标签路径（假设标签文件名与图像相同，扩展名为.png）
        label_name = img_name.rsplit('.', 1)[0] + '.png'
        label_path = os.path.join(self.lab_dir, label_name)
        
        # 加载图像和标签
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 应用数据增强
        if self.augment and self.mode == 'train':
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image_transformed = self.transform(image=image)['image']
            mask_transformed = self.mask_transform(image=mask)['image']
            image = image_transformed
            mask = mask_transformed
        
        # 二值化标签
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'name': img_name,
            'path': img_path
        }


def create_dataloaders(data_config, batch_size=8, num_workers=4):
    """创建训练和验证数据加载器"""
    
    # 训练集
    train_dataset = DeepCrackDataset(
        img_dir=data_config['train_img_dir'],
        lab_dir=data_config['train_lab_dir'],
        image_size=data_config.get('image_size', 224),
        mode='train',
        augment=data_config.get('augment', True)
    )
    
    # 验证集
    val_dataset = DeepCrackDataset(
        img_dir=data_config['val_img_dir'],
        lab_dir=data_config['val_lab_dir'],
        image_size=data_config.get('image_size', 224),
        mode='val',
        augment=False
    )
    
    # 测试集
    test_dataset = DeepCrackDataset(
        img_dir=data_config['test_img_dir'],
        lab_dir=data_config['test_lab_dir'],
        image_size=data_config.get('image_size', 224),
        mode='test',
        augment=False
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader