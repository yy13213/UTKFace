#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征降维与KDE计算模块
实现PCA降维(512→10维)和高斯核密度估计
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class PCAReducer:
    """PCA降维器"""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """拟合PCA并降维"""
        print(f"🔍 开始PCA降维: {features.shape[1]} → {self.n_components} 维")
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # PCA降维
        features_reduced = self.pca.fit_transform(features_scaled)
        
        self.fitted = True
        
        # 输出降维效果
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"✅ PCA降维完成:")
        print(f"   各主成分方差解释率: {explained_variance_ratio[:5].round(3)}")
        print(f"   累计方差解释率: {cumulative_variance[self.n_components-1]:.3f}")
        print(f"   降维后特征形状: {features_reduced.shape}")
        
        return features_reduced
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """对新数据进行降维"""
        if not self.fitted:
            raise ValueError("PCA未训练，请先调用fit_transform()")
        
        features_scaled = self.scaler.transform(features)
        return self.pca.transform(features_scaled)
    
    def get_component_info(self) -> Dict:
        """获取主成分分析信息"""
        if not self.fitted:
            return {}
        
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_,
            'n_components': self.n_components
        }

class KDECalculator:
    """核密度估计计算器"""
    
    def __init__(self, kernel: str = 'gaussian'):
        self.kernel = kernel
        self.kde = None
        self.bandwidth = None
        self.fitted = False
        
    def fit(self, features: np.ndarray, bandwidth: Optional[float] = None) -> float:
        """
        拟合KDE模型
        
        Args:
            features: 输入特征矩阵
            bandwidth: 带宽参数，如果为None则自动优化
            
        Returns:
            最优带宽值
        """
        print("🔍 开始KDE核密度估计...")
        
        if bandwidth is None:
            # 使用网格搜索优化带宽
            print("   正在优化带宽参数...")
            bandwidth = self._optimize_bandwidth(features)
        
        self.bandwidth = bandwidth
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth)
        self.kde.fit(features)
        self.fitted = True
        
        print(f"✅ KDE拟合完成:")
        print(f"   核函数: {self.kernel}")
        print(f"   最优带宽: {bandwidth:.4f}")
        print(f"   样本数量: {features.shape[0]}")
        
        return bandwidth
    
    def _optimize_bandwidth(self, features: np.ndarray) -> float:
        """优化带宽参数"""
        # 使用网格搜索优化带宽
        bandwidths = np.logspace(-2, 1, 20)  # 0.01 到 10
        
        grid = GridSearchCV(
            KernelDensity(kernel=self.kernel),
            {'bandwidth': bandwidths},
            cv=3,  # 3折交叉验证
            n_jobs=-1,
            verbose=0
        )
        
        grid.fit(features)
        return grid.best_params_['bandwidth']
    
    def compute_densities(self, features: np.ndarray) -> np.ndarray:
        """计算样本密度值"""
        if not self.fitted:
            raise ValueError("KDE未训练，请先调用fit()")
        
        # 计算对数密度并转换为密度
        log_densities = self.kde.score_samples(features)
        densities = np.exp(log_densities)
        
        print(f"✅ 密度计算完成:")
        print(f"   密度范围: {densities.min():.6f} - {densities.max():.6f}")
        print(f"   平均密度: {densities.mean():.6f}")
        
        return densities

class FeatureKDEAnalyzer:
    """特征降维与KDE分析的完整流程"""
    
    def __init__(self, n_components: int = 10):
        self.pca_reducer = PCAReducer(n_components)
        self.kde_calculator = KDECalculator()
        
    def analyze(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的特征降维与KDE分析
        
        Returns:
            reduced_features: 降维后的特征
            kde_densities: KDE密度值
        """
        # PCA降维
        reduced_features = self.pca_reducer.fit_transform(features)
        
        # KDE计算
        self.kde_calculator.fit(reduced_features)
        kde_densities = self.kde_calculator.compute_densities(reduced_features)
        
        return reduced_features, kde_densities
    
    def plot_pca_analysis(self, save_path: Optional[str] = None):
        """绘制PCA分析结果"""
        pca_info = self.pca_reducer.get_component_info()
        if not pca_info:
            print("⚠️ PCA未训练，无法绘制分析图")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('PCA降维分析', fontsize=16, fontweight='bold')
        
        # 方差解释率
        n_comp = len(pca_info['explained_variance_ratio'])
        axes[0].bar(range(1, n_comp+1), pca_info['explained_variance_ratio'])
        axes[0].set_title('各主成分方差解释率')
        axes[0].set_xlabel('主成分')
        axes[0].set_ylabel('方差解释率')
        axes[0].grid(True, alpha=0.3)
        
        # 累计方差解释率
        axes[1].plot(range(1, n_comp+1), pca_info['cumulative_variance_ratio'], 'o-')
        axes[1].axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='85%阈值')
        axes[1].set_title('累计方差解释率')
        axes[1].set_xlabel('主成分数量')
        axes[1].set_ylabel('累计方差解释率')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 PCA分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_kde_analysis(self, reduced_features: np.ndarray, kde_densities: np.ndarray, 
                         save_path: Optional[str] = None):
        """绘制KDE分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('KDE核密度估计分析', fontsize=16, fontweight='bold')
        
        # 密度分布直方图
        axes[0, 0].hist(kde_densities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('KDE密度分布')
        axes[0, 0].set_xlabel('密度值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 密度vs样本索引
        axes[0, 1].plot(kde_densities, alpha=0.6)
        axes[0, 1].set_title('密度值序列')
        axes[0, 1].set_xlabel('样本索引')
        axes[0, 1].set_ylabel('密度值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 前两个主成分的散点图（按密度着色）
        if reduced_features.shape[1] >= 2:
            scatter = axes[1, 0].scatter(reduced_features[:, 0], reduced_features[:, 1], 
                                       c=kde_densities, cmap='viridis', alpha=0.6, s=1)
            axes[1, 0].set_title('前两个主成分分布（按密度着色）')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[1, 0], label='KDE密度')
        
        # 密度箱线图
        axes[1, 1].boxplot(kde_densities, vert=True)
        axes[1, 1].set_title('密度分布箱线图')
        axes[1, 1].set_ylabel('密度值')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 KDE分析图已保存到: {save_path}")
        
        plt.show()
    
    def save_models(self, save_dir: str):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存PCA模型
        with open(os.path.join(save_dir, 'pca_reducer.pkl'), 'wb') as f:
            pickle.dump(self.pca_reducer, f)
        
        # 保存KDE模型
        with open(os.path.join(save_dir, 'kde_calculator.pkl'), 'wb') as f:
            pickle.dump(self.kde_calculator, f)
        
        print(f"📁 模型已保存到: {save_dir}")
    
    def load_models(self, save_dir: str):
        """加载模型"""
        # 加载PCA模型
        with open(os.path.join(save_dir, 'pca_reducer.pkl'), 'rb') as f:
            self.pca_reducer = pickle.load(f)
        
        # 加载KDE模型
        with open(os.path.join(save_dir, 'kde_calculator.pkl'), 'rb') as f:
            self.kde_calculator = pickle.load(f)
        
        print(f"📁 模型已从 {save_dir} 加载")

def run_kde_analysis(features: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行完整的特征降维与KDE分析
    
    Args:
        features: 输入特征矩阵 (N, 512)
        n_components: PCA主成分数量
        
    Returns:
        reduced_features: 降维后特征 (N, n_components)
        kde_densities: KDE密度值 (N,)
    """
    print(f"🚀 开始任务5：特征降维与KDE计算")
    print(f"   输入特征: {features.shape}")
    print(f"   目标维度: {n_components}")
    
    # 创建分析器
    analyzer = FeatureKDEAnalyzer(n_components)
    
    # 执行分析
    reduced_features, kde_densities = analyzer.analyze(features)
    
    # 绘制分析结果
    analyzer.plot_pca_analysis('results/plots/pca_analysis.png')
    analyzer.plot_kde_analysis(reduced_features, kde_densities, 'results/plots/kde_analysis.png')
    
    # 保存模型
    analyzer.save_models('models/')
    
    # 保存结果
    results_df = pd.DataFrame({
        'kde_density': kde_densities
    })
    results_df.to_csv('results/kde_densities.csv', index=False)
    
    # 保存降维后特征
    np.save('results/reduced_features.npy', reduced_features)
    
    print(f"✅ 任务5完成！")
    print(f"   降维后特征: {reduced_features.shape}")
    print(f"   KDE密度范围: {kde_densities.min():.6f} - {kde_densities.max():.6f}")
    print(f"   结果已保存到: results/")
    
    return reduced_features, kde_densities

if __name__ == "__main__":
    # 测试代码
    print("🧪 特征降维与KDE计算模块测试...")
    
    # 生成测试数据
    np.random.seed(42)
    test_features = np.random.randn(1000, 512)
    
    print("📝 使用随机数据进行测试")
    reduced_features, kde_densities = run_kde_analysis(test_features)
    
    print("✅ 测试完成！")
    print("📝 实际使用时请传入真实的特征矩阵") 