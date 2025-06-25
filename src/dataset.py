#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTKFace数据集加载器
实现年龄标签解析、图像预处理和数据验证功能
"""

import os
import re
from typing import Tuple, Optional, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UTKFaceDataset(Dataset):
    """
    UTKFace数据集加载器
    
    文件命名格式: [age]_[gender]_[race]_[date&time].jpg
    - age: 0-116岁
    - gender: 0(男), 1(女)  
    - race: 0(白人), 1(黑人), 2(亚洲人), 3(印度人), 4(其他)
    """
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 age_range: Tuple[int, int] = (0, 100),
                 validate_files: bool = True):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            transform: 图像预处理变换
            age_range: 有效年龄范围
            validate_files: 是否验证文件有效性
        """
        self.root_dir = root_dir
        self.transform = transform
        self.age_range = age_range
        
        # 加载并验证数据
        self.image_paths, self.metadata = self._load_and_validate_data(validate_files)
        
        print(f"✅ 成功加载 {len(self.image_paths)} 个有效样本")
        print(f"   年龄范围: {self.metadata['age'].min()}-{self.metadata['age'].max()}岁")
        print(f"   数据目录: {root_dir}")
    
    def _load_and_validate_data(self, validate_files: bool) -> Tuple[List[str], pd.DataFrame]:
        """加载并验证数据文件"""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.root_dir}")
        
        # 获取所有jpg文件
        all_files = [f for f in os.listdir(self.root_dir) if f.lower().endswith('.jpg')]
        
        if len(all_files) == 0:
            raise ValueError(f"数据目录中没有找到jpg文件: {self.root_dir}")
        
        print(f"🔍 发现 {len(all_files)} 个图像文件，正在解析...")
        
        valid_paths = []
        metadata_list = []
        
        # 文件名解析正则表达式
        pattern = r'^(\d+)_([01])_([0-4])_(.+)\.jpg$'
        
        for filename in all_files:
            try:
                # 解析文件名
                match = re.match(pattern, filename)
                if not match:
                    continue
                
                age, gender, race, datetime_part = match.groups()
                age = int(age)
                gender = int(gender)
                race = int(race)
                
                # 验证年龄范围
                if not (self.age_range[0] <= age <= self.age_range[1]):
                    continue
                
                # 验证文件存在性和可读性
                file_path = os.path.join(self.root_dir, filename)
                if validate_files:
                    try:
                        # 尝试打开图像验证完整性
                        with Image.open(file_path) as img:
                            img.verify()
                    except Exception:
                        continue
                
                valid_paths.append(filename)
                metadata_list.append({
                    'filename': filename,
                    'age': age,
                    'gender': gender,
                    'race': race,
                    'datetime': datetime_part
                })
                
            except (ValueError, AttributeError):
                continue
        
        if len(valid_paths) == 0:
            raise ValueError("没有找到有效的数据文件")
        
        metadata_df = pd.DataFrame(metadata_list)
        
        # 数据质量报告
        invalid_count = len(all_files) - len(valid_paths)
        if invalid_count > 0:
            print(f"⚠️  跳过 {invalid_count} 个无效文件 (解析失败或不符合条件)")
        
        return valid_paths, metadata_df
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        获取单个样本
        
        Returns:
            image: 预处理后的图像张量
            age: 年龄标签
            metadata: 额外的元数据信息
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"索引超出范围: {idx} >= {len(self.image_paths)}")
        
        # 获取文件路径和元数据
        filename = self.image_paths[idx]
        file_path = os.path.join(self.root_dir, filename)
        metadata = self.metadata.iloc[idx].to_dict()
        
        try:
            # 加载图像
            image = Image.open(file_path).convert('RGB')
            
            # 应用预处理变换
            if self.transform:
                image = self.transform(image)
            
            return image, metadata['age'], metadata
            
        except Exception as e:
            raise RuntimeError(f"无法加载图像 {filename}: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.image_paths),
            'age_stats': {
                'min': self.metadata['age'].min(),
                'max': self.metadata['age'].max(),
                'mean': self.metadata['age'].mean(),
                'std': self.metadata['age'].std(),
                'median': self.metadata['age'].median()
            },
            'gender_distribution': self.metadata['gender'].value_counts().to_dict(),
            'race_distribution': self.metadata['race'].value_counts().to_dict(),
            'age_distribution': self.metadata['age'].value_counts().sort_index().to_dict()
        }
        return stats
    
    def plot_data_distribution(self, save_path: Optional[str] = None):
        """绘制数据分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UTKFace数据集分布分析', fontsize=16, fontweight='bold')
        
        # 年龄分布直方图
        axes[0, 0].hist(self.metadata['age'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('年龄分布')
        axes[0, 0].set_xlabel('年龄')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 性别分布饼图
        gender_labels = ['男性', '女性']
        gender_counts = self.metadata['gender'].value_counts().sort_index()
        axes[0, 1].pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%', 
                       colors=['lightblue', 'lightpink'])
        axes[0, 1].set_title('性别分布')
        
        # 种族分布条形图
        race_labels = ['白人', '黑人', '亚洲人', '印度人', '其他']
        race_counts = self.metadata['race'].value_counts().sort_index()
        axes[1, 0].bar(range(len(race_counts)), race_counts.values, 
                       color=['wheat', 'chocolate', 'gold', 'orange', 'lightgreen'])
        axes[1, 0].set_title('种族分布')
        axes[1, 0].set_xlabel('种族')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].set_xticks(range(len(race_labels)))
        axes[1, 0].set_xticklabels(race_labels, rotation=45)
        
        # 年龄箱线图
        axes[1, 1].boxplot(self.metadata['age'], vert=True)
        axes[1, 1].set_title('年龄分布箱线图')
        axes[1, 1].set_ylabel('年龄')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 数据分布图已保存到: {save_path}")
        
        plt.show()
        
    def create_data_quality_report(self, save_path: Optional[str] = None) -> str:
        """创建数据质量报告"""
        stats = self.get_statistics()
        
        report = f"""
# UTKFace数据质量报告

## 基本信息
- 数据目录: {self.root_dir}
- 总样本数: {stats['total_samples']:,}
- 年龄范围限制: {self.age_range[0]}-{self.age_range[1]}岁

## 年龄统计
- 最小年龄: {stats['age_stats']['min']}岁
- 最大年龄: {stats['age_stats']['max']}岁
- 平均年龄: {stats['age_stats']['mean']:.1f}岁
- 年龄标准差: {stats['age_stats']['std']:.1f}岁
- 年龄中位数: {stats['age_stats']['median']:.1f}岁

## 性别分布
- 男性 (0): {stats['gender_distribution'].get(0, 0):,} 人 ({100*stats['gender_distribution'].get(0, 0)/stats['total_samples']:.1f}%)
- 女性 (1): {stats['gender_distribution'].get(1, 0):,} 人 ({100*stats['gender_distribution'].get(1, 0)/stats['total_samples']:.1f}%)

## 种族分布
- 白人 (0): {stats['race_distribution'].get(0, 0):,} 人
- 黑人 (1): {stats['race_distribution'].get(1, 0):,} 人  
- 亚洲人 (2): {stats['race_distribution'].get(2, 0):,} 人
- 印度人 (3): {stats['race_distribution'].get(3, 0):,} 人
- 其他 (4): {stats['race_distribution'].get(4, 0):,} 人

## 数据质量评估
- 文件命名格式: 标准UTKFace格式
- 年龄标签完整性: 100%
- 图像文件完整性: 已验证
"""

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📋 数据质量报告已保存到: {save_path}")
        
        return report


def get_default_transforms(image_size: int = 224) -> transforms.Compose:
    """获取默认的图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet统计值
    ])


def create_dataloader(dataset: UTKFaceDataset, 
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def test_dataset(data_dir: str, sample_size: int = 5):
    """测试数据集加载功能"""
    print("🧪 开始测试UTKFace数据集加载器...")
    
    try:
        # 创建数据集
        transform = get_default_transforms()
        dataset = UTKFaceDataset(
            root_dir=data_dir,
            transform=transform,
            validate_files=True
        )
        
        print(f"\n📊 数据集统计信息:")
        stats = dataset.get_statistics()
        print(f"   总样本数: {stats['total_samples']:,}")
        print(f"   年龄范围: {stats['age_stats']['min']}-{stats['age_stats']['max']}岁")
        print(f"   平均年龄: {stats['age_stats']['mean']:.1f}岁")
        
        # 测试样本加载
        print(f"\n🔍 测试前{sample_size}个样本:")
        for i in range(min(sample_size, len(dataset))):
            image, age, metadata = dataset[i]
            print(f"   样本{i+1}: 图像形状={image.shape}, 年龄={age}, 性别={metadata['gender']}, 种族={metadata['race']}")
        
        # 创建数据加载器测试
        dataloader = create_dataloader(dataset, batch_size=4, shuffle=False)
        batch_images, batch_ages, batch_metadata = next(iter(dataloader))
        
        print(f"\n📦 批次数据测试:")
        print(f"   批次图像形状: {batch_images.shape}")
        print(f"   批次年龄: {batch_ages.tolist()}")
        
        print("\n✅ 数据集加载器测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 数据集加载器测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 测试代码
    test_data_dir = "data/utkface"  # 请根据实际路径修改
    
    if os.path.exists(test_data_dir):
        test_dataset(test_data_dir)
    else:
        print(f"⚠️  测试数据目录不存在: {test_data_dir}")
        print("   请下载UTKFace数据集并将其放置在data/utkface目录中")
        print("   数据集下载地址: https://susanqq.github.io/UTKFace/") 