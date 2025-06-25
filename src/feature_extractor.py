#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取与年龄预测模块
使用预训练ResNet18提取512维特征，训练回归器预测年龄并计算MAE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    """ResNet18特征提取器"""
    
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 加载预训练ResNet18
        resnet = models.resnet18(pretrained=True)
        # 移除最后的分类层，保留特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 输出512维特征
        return x

class AgeRegressor(nn.Module):
    """简单的年龄回归器"""
    
    def __init__(self, input_dim=512):
        super(AgeRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.regressor(x).squeeze()

class FeatureAgePredictor:
    """特征提取与年龄预测的完整流程"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.age_regressor = AgeRegressor().to(self.device)
        
        # 冻结特征提取器参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        print(f"✅ 模型已初始化，使用设备: {self.device}")
    
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量提取特征
        
        Returns:
            features: (N, 512) 特征矩阵
            ages: (N,) 年龄标签
        """
        self.feature_extractor.eval()
        
        all_features = []
        all_ages = []
        
        print("🔍 开始特征提取...")
        with torch.no_grad():
            for batch_images, batch_ages, _ in tqdm(dataloader, desc="特征提取"):
                batch_images = batch_images.to(self.device)
                
                # 提取特征
                features = self.feature_extractor(batch_images)
                
                all_features.append(features.cpu().numpy())
                all_ages.extend(batch_ages.numpy())
        
        features = np.vstack(all_features)
        ages = np.array(all_ages)
        
        print(f"✅ 特征提取完成: {features.shape[0]}个样本，{features.shape[1]}维特征")
        return features, ages
    
    def train_age_regressor(self, features: np.ndarray, ages: np.ndarray, 
                           test_size: float = 0.2, epochs: int = 50) -> Dict:
        """
        训练年龄回归器
        
        Returns:
            训练结果字典，包含MAE、R²等指标
        """
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, ages, test_size=test_size, random_state=42
        )
        
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # 训练设置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.age_regressor.parameters(), lr=0.001)
        
        # 训练循环
        self.age_regressor.train()
        train_losses = []
        
        print(f"🏋️ 开始训练年龄回归器...")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.age_regressor(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 评估模型
        self.age_regressor.eval()
        with torch.no_grad():
            train_pred = self.age_regressor(X_train_tensor).cpu().numpy()
            test_pred = self.age_regressor(X_test_tensor).cpu().numpy()
        
        # 计算指标
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_losses': train_losses,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"✅ 训练完成！")
        print(f"   训练集 MAE: {train_mae:.2f}岁, R²: {train_r2:.3f}")
        print(f"   测试集 MAE: {test_mae:.2f}岁, R²: {test_r2:.3f}")
        
        return results
    
    def compute_all_maes(self, features: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """
        计算所有样本的MAE
        
        Returns:
            mae_values: (N,) 每个样本的绝对误差
        """
        self.age_regressor.eval()
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            predictions = self.age_regressor(features_tensor).cpu().numpy()
        
        # 计算每个样本的绝对误差
        mae_values = np.abs(predictions - ages)
        
        print(f"✅ MAE计算完成，平均MAE: {mae_values.mean():.2f}岁")
        return mae_values
    
    def save_models(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.feature_extractor.state_dict(), 
                  os.path.join(save_dir, 'feature_extractor.pth'))
        torch.save(self.age_regressor.state_dict(), 
                  os.path.join(save_dir, 'age_regressor.pth'))
        
        print(f"📁 模型已保存到: {save_dir}")
    
    def load_models(self, save_dir: str):
        """加载模型"""
        self.feature_extractor.load_state_dict(
            torch.load(os.path.join(save_dir, 'feature_extractor.pth')))
        self.age_regressor.load_state_dict(
            torch.load(os.path.join(save_dir, 'age_regressor.pth')))
        
        print(f"📁 模型已从 {save_dir} 加载")

def plot_training_results(results: Dict, save_path: Optional[str] = None):
    """绘制训练结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('年龄预测模型训练结果', fontsize=16, fontweight='bold')
    
    # 训练损失曲线
    axes[0, 0].plot(results['train_losses'])
    axes[0, 0].set_title('训练损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练集预测vs真实
    axes[0, 1].scatter(results['y_train'], results['train_predictions'], alpha=0.5, s=1)
    axes[0, 1].plot([0, 100], [0, 100], 'r--', alpha=0.8)
    axes[0, 1].set_title(f'训练集预测效果 (MAE: {results["train_mae"]:.2f})')
    axes[0, 1].set_xlabel('真实年龄')
    axes[0, 1].set_ylabel('预测年龄')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 测试集预测vs真实
    axes[1, 0].scatter(results['y_test'], results['test_predictions'], alpha=0.6, s=1)
    axes[1, 0].plot([0, 100], [0, 100], 'r--', alpha=0.8)
    axes[1, 0].set_title(f'测试集预测效果 (MAE: {results["test_mae"]:.2f})')
    axes[1, 0].set_xlabel('真实年龄')
    axes[1, 0].set_ylabel('预测年龄')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差分布
    train_errors = np.abs(results['y_train'] - results['train_predictions'])
    test_errors = np.abs(results['y_test'] - results['test_predictions'])
    
    axes[1, 1].hist(train_errors, bins=30, alpha=0.5, label='训练集', density=True)
    axes[1, 1].hist(test_errors, bins=30, alpha=0.5, label='测试集', density=True)
    axes[1, 1].set_title('预测误差分布')
    axes[1, 1].set_xlabel('绝对误差 (岁)')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练结果图已保存到: {save_path}")
    
    plt.show()

def run_feature_extraction_and_prediction(dataset, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    运行完整的特征提取与年龄预测流程
    
    Returns:
        features: (N, 512) 特征矩阵
        ages: (N,) 年龄标签  
        mae_values: (N,) 每个样本的MAE
    """
    from dataset import create_dataloader
    
    # 创建数据加载器
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    predictor = FeatureAgePredictor()
    
    # 特征提取
    features, ages = predictor.extract_features(dataloader)
    
    # 训练年龄回归器
    training_results = predictor.train_age_regressor(features, ages)
    
    # 绘制训练结果
    plot_training_results(training_results, 'results/plots/age_prediction_results.png')
    
    # 计算所有样本的MAE
    mae_values = predictor.compute_all_maes(features, ages)
    
    # 保存模型
    predictor.save_models('models/')
    
    # 保存结果
    results_df = pd.DataFrame({
        'age': ages,
        'mae': mae_values
    })
    results_df.to_csv('results/features_and_mae.csv', index=False)
    
    # 保存特征矩阵
    np.save('results/features.npy', features)
    
    print(f"✅ 任务4完成！")
    print(f"   特征矩阵: {features.shape}")
    print(f"   平均MAE: {mae_values.mean():.2f}岁")
    print(f"   结果已保存到: results/")
    
    return features, ages, mae_values

if __name__ == "__main__":
    # 测试代码
    print("🧪 特征提取与年龄预测模块测试...")
    
    # 这里需要实际的数据集来测试
    # test_data_dir = "data/utkface"
    # if os.path.exists(test_data_dir):
    #     from dataset import UTKFaceDataset, get_default_transforms
    #     dataset = UTKFaceDataset(test_data_dir, transform=get_default_transforms())
    #     features, ages, mae_values = run_feature_extraction_and_prediction(dataset)
    # else:
    #     print("⚠️  需要UTKFace数据集来运行测试")
    
    print("📝 请确保有UTKFace数据集后调用 run_feature_extraction_and_prediction() 函数") 