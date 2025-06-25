#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDE-MAE关系分析模块
分析核密度估计值与预测误差的相关性
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class KDEMAECorrelationAnalyzer:
    """KDE与MAE相关性分析器"""
    
    def __init__(self):
        self.correlation_results = {}
        
    def analyze_correlation(self, kde_densities: np.ndarray, mae_values: np.ndarray) -> Dict:
        """
        分析KDE密度值与MAE的相关性
        
        Args:
            kde_densities: KDE密度值
            mae_values: MAE值
            
        Returns:
            相关性分析结果字典
        """
        print("🔍 开始KDE-MAE相关性分析...")
        
        # 基本统计
        print(f"   KDE密度范围: {kde_densities.min():.6f} - {kde_densities.max():.6f}")
        print(f"   MAE范围: {mae_values.min():.2f} - {mae_values.max():.2f}")
        
        # 皮尔逊相关系数
        pearson_corr, pearson_p = stats.pearsonr(kde_densities, mae_values)
        
        # 斯皮尔曼等级相关系数（非线性关系）
        spearman_corr, spearman_p = stats.spearmanr(kde_densities, mae_values)
        
        # 肯德尔τ相关系数
        kendall_corr, kendall_p = stats.kendalltau(kde_densities, mae_values)
        
        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(kde_densities, mae_values)
        
        results = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr, 
            'spearman_p_value': spearman_p,
            'kendall_correlation': kendall_corr,
            'kendall_p_value': kendall_p,
            'linear_regression': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            },
            'data_stats': {
                'kde_mean': kde_densities.mean(),
                'kde_std': kde_densities.std(),
                'mae_mean': mae_values.mean(),
                'mae_std': mae_values.std(),
                'sample_size': len(kde_densities)
            }
        }
        
        self.correlation_results = results
        
        print(f"✅ 相关性分析完成:")
        print(f"   皮尔逊相关系数: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"   斯皮尔曼相关系数: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"   线性拟合R²: {r_value**2:.4f}")
        
        # 判断相关性强度
        abs_corr = abs(pearson_corr)
        if abs_corr > 0.7:
            strength = "强"
        elif abs_corr > 0.3:
            strength = "中等"
        else:
            strength = "弱"
        
        direction = "负" if pearson_corr < 0 else "正"
        print(f"   相关性强度: {strength}{direction}相关")
        
        return results
    
    def plot_correlation_analysis(self, kde_densities: np.ndarray, mae_values: np.ndarray, 
                                save_path: Optional[str] = None):
        """绘制KDE-MAE相关性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KDE-MAE相关性分析', fontsize=16, fontweight='bold')
        
        # 散点图 + 趋势线
        axes[0, 0].scatter(kde_densities, mae_values, alpha=0.6, s=2, color='blue')
        
        # 添加线性回归线
        if self.correlation_results:
            lr_results = self.correlation_results['linear_regression']
            x_line = np.linspace(kde_densities.min(), kde_densities.max(), 100)
            y_line = lr_results['slope'] * x_line + lr_results['intercept']
            axes[0, 0].plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2, 
                           label=f'线性拟合 (R²={lr_results["r_squared"]:.3f})')
        
        axes[0, 0].set_xlabel('KDE密度值')
        axes[0, 0].set_ylabel('MAE (岁)')
        axes[0, 0].set_title('KDE密度 vs MAE散点图')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 六边形密度图（处理密集点）
        axes[0, 1].hexbin(kde_densities, mae_values, gridsize=30, cmap='Blues', alpha=0.8)
        axes[0, 1].set_xlabel('KDE密度值')
        axes[0, 1].set_ylabel('MAE (岁)')
        axes[0, 1].set_title('KDE-MAE密度分布图')
        
        # KDE密度分布直方图
        axes[1, 0].hist(kde_densities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_xlabel('KDE密度值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('KDE密度分布')
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE分布直方图
        axes[1, 1].hist(mae_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_xlabel('MAE (岁)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('MAE分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 相关性分析图已保存到: {save_path}")
        
        plt.show()
    
    def plot_detailed_correlation(self, kde_densities: np.ndarray, mae_values: np.ndarray,
                                save_path: Optional[str] = None):
        """绘制详细的相关性分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KDE-MAE详细相关性分析', fontsize=16, fontweight='bold')
        
        # 1. 散点图 + 多项式拟合
        axes[0, 0].scatter(kde_densities, mae_values, alpha=0.5, s=1)
        
        # 线性拟合
        z_linear = np.polyfit(kde_densities, mae_values, 1)
        p_linear = np.poly1d(z_linear)
        x_fit = np.linspace(kde_densities.min(), kde_densities.max(), 100)
        axes[0, 0].plot(x_fit, p_linear(x_fit), "r--", alpha=0.8, label='线性拟合')
        
        # 二次拟合
        z_quad = np.polyfit(kde_densities, mae_values, 2)
        p_quad = np.poly1d(z_quad)
        axes[0, 0].plot(x_fit, p_quad(x_fit), "g--", alpha=0.8, label='二次拟合')
        
        axes[0, 0].set_xlabel('KDE密度值')
        axes[0, 0].set_ylabel('MAE (岁)')
        axes[0, 0].set_title('散点图 + 拟合曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 分位数分析
        # 将KDE分为若干组，分析每组的MAE分布
        n_bins = 10
        kde_bins = np.percentile(kde_densities, np.linspace(0, 100, n_bins+1))
        bin_centers = []
        bin_mae_means = []
        bin_mae_stds = []
        
        for i in range(n_bins):
            mask = (kde_densities >= kde_bins[i]) & (kde_densities < kde_bins[i+1])
            if np.sum(mask) > 0:
                bin_centers.append((kde_bins[i] + kde_bins[i+1]) / 2)
                bin_mae_means.append(mae_values[mask].mean())
                bin_mae_stds.append(mae_values[mask].std())
        
        axes[0, 1].errorbar(bin_centers, bin_mae_means, yerr=bin_mae_stds, 
                           fmt='o-', capsize=5, capthick=2)
        axes[0, 1].set_xlabel('KDE密度值 (分组中心)')
        axes[0, 1].set_ylabel('平均MAE (岁)')
        axes[0, 1].set_title('分组分析：KDE vs 平均MAE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 残差分析
        if self.correlation_results:
            lr_results = self.correlation_results['linear_regression']
            predicted_mae = lr_results['slope'] * kde_densities + lr_results['intercept']
            residuals = mae_values - predicted_mae
            
            axes[0, 2].scatter(predicted_mae, residuals, alpha=0.5, s=1)
            axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[0, 2].set_xlabel('预测MAE')
            axes[0, 2].set_ylabel('残差')
            axes[0, 2].set_title('线性拟合残差图')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 相关系数比较
        if self.correlation_results:
            corr_names = ['Pearson', 'Spearman', 'Kendall']
            corr_values = [
                self.correlation_results['pearson_correlation'],
                self.correlation_results['spearman_correlation'],
                self.correlation_results['kendall_correlation']
            ]
            
            bars = axes[1, 0].bar(corr_names, corr_values, alpha=0.7, 
                                 color=['blue', 'green', 'orange'])
            axes[1, 0].set_ylabel('相关系数')
            axes[1, 0].set_title('不同相关系数比较')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, corr_values):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # 5. 密度等高线图
        try:
            sns.kdeplot(x=kde_densities, y=mae_values, ax=axes[1, 1], levels=10)
            axes[1, 1].set_xlabel('KDE密度值')
            axes[1, 1].set_ylabel('MAE (岁)')
            axes[1, 1].set_title('二维密度等高线图')
        except Exception:
            axes[1, 1].scatter(kde_densities, mae_values, alpha=0.5, s=1)
            axes[1, 1].set_xlabel('KDE密度值')
            axes[1, 1].set_ylabel('MAE (岁)')
            axes[1, 1].set_title('散点图（备用）')
        
        # 6. 统计信息表
        if self.correlation_results:
            stats_text = f"""
相关性分析结果：

皮尔逊相关系数: {self.correlation_results['pearson_correlation']:.4f}
p值: {self.correlation_results['pearson_p_value']:.4f}

斯皮尔曼相关系数: {self.correlation_results['spearman_correlation']:.4f}
p值: {self.correlation_results['spearman_p_value']:.4f}

线性拟合 R²: {self.correlation_results['linear_regression']['r_squared']:.4f}

样本数量: {self.correlation_results['data_stats']['sample_size']:,}
            """
            
            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
            axes[1, 2].set_title('统计结果摘要')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 详细相关性分析图已保存到: {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """生成相关性分析报告"""
        if not self.correlation_results:
            return "尚未进行相关性分析"
        
        results = self.correlation_results
        
        # 判断相关性强度和显著性
        pearson_corr = results['pearson_correlation']
        pearson_p = results['pearson_p_value']
        
        abs_corr = abs(pearson_corr)
        if abs_corr > 0.7:
            strength = "强"
        elif abs_corr > 0.5:
            strength = "中等偏强"
        elif abs_corr > 0.3:
            strength = "中等"
        elif abs_corr > 0.1:
            strength = "弱"
        else:
            strength = "极弱"
        
        direction = "负" if pearson_corr < 0 else "正"
        significance = "显著" if pearson_p < 0.05 else "不显著"
        
        report = f"""
# KDE-MAE相关性分析报告

## 核心发现
- **相关性强度**: {strength}{direction}相关
- **统计显著性**: {significance} (p = {pearson_p:.4f})
- **皮尔逊相关系数**: {pearson_corr:.4f}
- **线性关系解释力**: R² = {results['linear_regression']['r_squared']:.4f}

## 详细统计结果

### 相关系数分析
- 皮尔逊相关系数: {results['pearson_correlation']:.4f} (p = {results['pearson_p_value']:.4f})
- 斯皮尔曼等级相关: {results['spearman_correlation']:.4f} (p = {results['spearman_p_value']:.4f})  
- 肯德尔τ相关: {results['kendall_correlation']:.4f} (p = {results['kendall_p_value']:.4f})

### 线性回归分析
- 斜率: {results['linear_regression']['slope']:.6f}
- 截距: {results['linear_regression']['intercept']:.6f}
- R²决定系数: {results['linear_regression']['r_squared']:.4f}
- p值: {results['linear_regression']['p_value']:.4f}
- 标准误差: {results['linear_regression']['std_error']:.6f}

### 数据统计
- 样本数量: {results['data_stats']['sample_size']:,}
- KDE密度均值: {results['data_stats']['kde_mean']:.6f}
- KDE密度标准差: {results['data_stats']['kde_std']:.6f}
- MAE均值: {results['data_stats']['mae_mean']:.2f}岁
- MAE标准差: {results['data_stats']['mae_std']:.2f}岁

## 结论解释

{'如果KDE密度越高，预测误差越小，说明模型在特征空间密集区域表现更好。' if pearson_corr < 0 else '如果KDE密度越高，预测误差越大，可能存在过拟合或数据质量问题。'}

相关性的{significance}性表明{'这种关系具有统计学意义。' if pearson_p < 0.05 else '这种关系可能是随机的。'}
        """
        
        return report

def run_correlation_analysis(kde_densities: np.ndarray, mae_values: np.ndarray) -> Dict:
    """
    运行完整的KDE-MAE相关性分析
    
    Args:
        kde_densities: KDE密度值
        mae_values: MAE值
        
    Returns:
        相关性分析结果
    """
    print(f"🚀 开始任务6：KDE-MAE关系分析")
    print(f"   样本数量: {len(kde_densities)}")
    
    # 创建分析器
    analyzer = KDEMAECorrelationAnalyzer()
    
    # 执行相关性分析
    results = analyzer.analyze_correlation(kde_densities, mae_values)
    
    # 绘制分析图表
    analyzer.plot_correlation_analysis(kde_densities, mae_values, 
                                     'results/plots/kde_mae_correlation.png')
    analyzer.plot_detailed_correlation(kde_densities, mae_values,
                                     'results/plots/kde_mae_detailed_analysis.png')
    
    # 生成并保存报告
    report = analyzer.generate_report()
    with open('results/kde_mae_correlation_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存结果数据
    results_df = pd.DataFrame({
        'kde_density': kde_densities,
        'mae_value': mae_values
    })
    results_df.to_csv('results/kde_mae_data.csv', index=False)
    
    print(f"✅ 任务6完成！")
    print(f"   相关性系数: {results['pearson_correlation']:.4f}")
    print(f"   统计显著性: {'显著' if results['pearson_p_value'] < 0.05 else '不显著'}")
    print(f"   报告已保存到: results/kde_mae_correlation_report.md")
    
    return results

if __name__ == "__main__":
    # 测试代码
    print("🧪 KDE-MAE相关性分析模块测试...")
    
    # 生成测试数据
    np.random.seed(42)
    kde_densities = np.random.exponential(0.01, 1000)  # 模拟KDE密度
    mae_values = 5 + 2 * np.log(1/kde_densities) + np.random.normal(0, 1, 1000)  # 模拟负相关关系
    
    print("📝 使用模拟数据进行测试")
    results = run_correlation_analysis(kde_densities, mae_values)
    
    print("✅ 测试完成！")
    print("📝 实际使用时请传入真实的KDE密度和MAE数据") 