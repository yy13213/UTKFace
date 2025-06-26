import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import torch
from typing import Tuple, Optional, List
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultTableGenerator:
    """结果表格生成器"""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """设置绘图样式"""
        sns.set_style("whitegrid")
        rcParams['figure.figsize'] = (15, 10)
        rcParams['font.size'] = 10
    
    def create_results_table(self, 
                           features: np.ndarray,
                           predictions: np.ndarray,
                           true_values: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           sample_indices: Optional[np.ndarray] = None,
                           max_samples: int = 50) -> pd.DataFrame:
        """
        创建结果表格
        
        Args:
            features: 特征数组 (n_samples, n_features)
            predictions: 预测值数组 (n_samples,)
            true_values: 真实值数组 (n_samples,)
            feature_names: 特征名称列表
            sample_indices: 样本索引
            max_samples: 最大显示样本数
            
        Returns:
            pd.DataFrame: 结果表格
        """
        n_samples, n_features = features.shape
        
        # 限制显示的样本数量
        if n_samples > max_samples:
            if sample_indices is None:
                # 随机选择样本，包括一些高误差和低误差的样本
                abs_errors = np.abs(predictions - true_values)
                
                # 选择误差最大的样本
                high_error_indices = np.argsort(abs_errors)[-max_samples//3:]
                # 选择误差最小的样本
                low_error_indices = np.argsort(abs_errors)[:max_samples//3]
                # 选择中等误差的样本
                remaining = max_samples - len(high_error_indices) - len(low_error_indices)
                mid_indices = np.random.choice(
                    n_samples, 
                    size=remaining, 
                    replace=False
                )
                
                sample_indices = np.concatenate([
                    low_error_indices, 
                    mid_indices, 
                    high_error_indices
                ])
            else:
                sample_indices = sample_indices[:max_samples]
            
            features = features[sample_indices]
            predictions = predictions[sample_indices]
            true_values = true_values[sample_indices]
        else:
            if sample_indices is None:
                sample_indices = np.arange(n_samples)
        
        # 计算绝对误差
        abs_errors = np.abs(predictions - true_values)
        
        # 创建特征名称
        if feature_names is None:
            feature_names = [f'特征_{i+1}' for i in range(n_features)]
        
        # 构建表格数据
        table_data = {}
        
        # 添加样本索引
        table_data['样本ID'] = sample_indices
        
        # 添加特征列
        for i, feature_name in enumerate(feature_names):
            table_data[feature_name] = features[:, i]
        
        # 添加预测结果列
        table_data['预测值'] = predictions
        table_data['真实值'] = true_values
        table_data['绝对误差'] = abs_errors
        table_data['相对误差(%)'] = (abs_errors / np.maximum(true_values, 1e-6)) * 100
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        return df
    
    def plot_results_table(self, 
                          df: pd.DataFrame,
                          save_path: str = 'results/plots/results_table.png',
                          title: str = 'UTKFace年龄预测结果详细表格') -> None:
        """
        绘制结果表格
        
        Args:
            df: 结果DataFrame
            save_path: 保存路径
            title: 图表标题
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 设置图形大小
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # 准备表格数据 - 只显示主要列
        display_columns = ['样本ID', '预测值', '真实值', '绝对误差', '相对误差(%)']
        
        # 如果有特征列，选择前几个重要特征
        feature_cols = [col for col in df.columns if col.startswith('特征') or col.startswith('PC')]
        if feature_cols:
            # 最多显示5个特征
            selected_features = feature_cols[:5]
            display_columns = ['样本ID'] + selected_features + ['预测值', '真实值', '绝对误差', '相对误差(%)']
        
        table_data = df[display_columns].round(3)
        
        # 创建颜色映射 - 根据绝对误差着色
        abs_errors = df['绝对误差'].values
        error_percentiles = np.percentile(abs_errors, [33, 67])
        
        # 为每行创建颜色，修正颜色数组的维度
        row_colors = []
        for error in abs_errors:
            if error <= error_percentiles[0]:
                row_color = '#d4edda'  # 绿色 - 低误差
            elif error <= error_percentiles[1]:
                row_color = '#fff3cd'  # 黄色 - 中等误差
            else:
                row_color = '#f8d7da'  # 红色 - 高误差
            
            # 为这一行的所有列创建相同颜色
            row_colors.append([row_color] * len(display_columns))
        
        # 绘制表格
        table = ax.table(
            cellText=table_data.values,
            colLabels=display_columns,
            cellLoc='center',
            loc='center',
            cellColours=row_colors
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # 设置表头样式
        for i in range(len(display_columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 设置标题
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#d4edda', label='低误差 (前33%)'),
            plt.Rectangle((0,0),1,1, facecolor='#fff3cd', label='中等误差 (33%-67%)'),
            plt.Rectangle((0,0),1,1, facecolor='#f8d7da', label='高误差 (后33%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # 添加统计信息
        stats_text = f"""统计信息:
        样本数量: {len(df)}
        平均绝对误差: {df['绝对误差'].mean():.3f}
        误差标准差: {df['绝对误差'].std():.3f}
        最大误差: {df['绝对误差'].max():.3f}
        最小误差: {df['绝对误差'].min():.3f}
        """
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 结果表格已保存到: {save_path}")
    
    def plot_error_distribution_table(self, 
                                    df: pd.DataFrame,
                                    save_path: str = 'results/plots/error_distribution_table.png') -> None:
        """
        绘制误差分布统计表格
        
        Args:
            df: 结果DataFrame
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. 误差分组统计表格
        abs_errors = df['绝对误差'].values
        
        # 定义误差区间
        error_bins = [0, 1, 2, 3, 5, 10, float('inf')]
        error_labels = ['0-1岁', '1-2岁', '2-3岁', '3-5岁', '5-10岁', '>10岁']
        
        # 统计每个区间的样本数
        error_counts = []
        error_percentages = []
        
        for i in range(len(error_bins)-1):
            mask = (abs_errors >= error_bins[i]) & (abs_errors < error_bins[i+1])
            count = np.sum(mask)
            percentage = count / len(abs_errors) * 100
            error_counts.append(count)
            error_percentages.append(percentage)
        
        # 创建误差分布表格
        error_stats = pd.DataFrame({
            '误差区间': error_labels,
            '样本数量': error_counts,
            '占比(%)': [f'{p:.1f}%' for p in error_percentages],
            '累计占比(%)': [f'{np.sum(error_percentages[:i+1]):.1f}%' for i in range(len(error_percentages))]
        })
        
        # 绘制误差分布表格
        table1 = ax1.table(
            cellText=error_stats.values,
            colLabels=error_stats.columns,
            cellLoc='center',
            loc='center'
        )
        
        table1.auto_set_font_size(False)
        table1.set_fontsize(12)
        table1.scale(1.2, 2)
        
        # 设置表头样式
        for i in range(len(error_stats.columns)):
            table1[(0, i)].set_facecolor('#2196F3')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.axis('off')
        ax1.set_title('误差分布统计表', fontsize=14, fontweight='bold', pad=20)
        
        # 2. 详细统计信息表格
        stats_data = {
            '统计指标': [
                '样本总数', '平均绝对误差', '误差标准差', '中位数误差',
                '最小误差', '最大误差', '25%分位数', '75%分位数',
                '平均相对误差(%)', 'RMSE'
            ],
            '数值': [
                len(df),
                f'{df["绝对误差"].mean():.3f}',
                f'{df["绝对误差"].std():.3f}',
                f'{df["绝对误差"].median():.3f}',
                f'{df["绝对误差"].min():.3f}',
                f'{df["绝对误差"].max():.3f}',
                f'{df["绝对误差"].quantile(0.25):.3f}',
                f'{df["绝对误差"].quantile(0.75):.3f}',
                f'{df["相对误差(%)"].mean():.1f}%',
                f'{np.sqrt(np.mean((df["预测值"] - df["真实值"])**2)):.3f}'
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        
        # 绘制统计信息表格
        table2 = ax2.table(
            cellText=stats_df.values,
            colLabels=stats_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table2.auto_set_font_size(False)
        table2.set_fontsize(12)
        table2.scale(1.2, 2)
        
        # 设置表头样式
        for i in range(len(stats_df.columns)):
            table2[(0, i)].set_facecolor('#FF9800')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.axis('off')
        ax2.set_title('详细统计信息', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('UTKFace年龄预测误差统计分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 误差分布表格已保存到: {save_path}")
    
    def export_results_to_csv(self, 
                             df: pd.DataFrame,
                             save_path: str = 'results/metrics/prediction_results.csv') -> None:
        """
        导出结果到CSV文件
        
        Args:
            df: 结果DataFrame
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 结果数据已导出到: {save_path}")
        print(f"📊 数据包含 {len(df)} 个样本，{len(df.columns)} 个字段")

def demo_create_results_table():
    """演示如何创建和可视化结果表格"""
    
    print("🎯 演示结果表格创建...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # 生成特征数据 (PCA降维后的特征)
    features = np.random.randn(n_samples, n_features)
    
    # 生成真实年龄 (18-80岁)
    true_ages = np.random.uniform(18, 80, n_samples)
    
    # 生成预测年龄 (添加一些误差)
    noise = np.random.normal(0, 3, n_samples)  # 3岁标准差的噪声
    predicted_ages = true_ages + noise
    
    # 确保预测值在合理范围内
    predicted_ages = np.clip(predicted_ages, 0, 100)
    
    # 创建特征名称
    feature_names = [f'PC{i+1}' for i in range(n_features)]
    
    # 创建结果表格生成器
    generator = ResultTableGenerator()
    
    # 创建结果表格
    results_df = generator.create_results_table(
        features=features,
        predictions=predicted_ages,
        true_values=true_ages,
        feature_names=feature_names,
        max_samples=30  # 显示30个样本
    )
    
    print(f"\n📋 结果表格预览:")
    print(results_df.head(10))
    
    # 绘制结果表格
    generator.plot_results_table(
        df=results_df,
        save_path='results/plots/06_results_table.png',
        title='UTKFace年龄预测结果详细表格 (演示数据)'
    )
    
    # 绘制误差分布表格
    generator.plot_error_distribution_table(
        df=results_df,
        save_path='results/plots/07_error_distribution_table.png'
    )
    
    # 导出CSV文件
    generator.export_results_to_csv(
        df=results_df,
        save_path='results/metrics/demo_prediction_results.csv'
    )
    
    print("\n📈 结果表格统计信息:")
    print(f"   - 样本数量: {len(results_df)}")
    print(f"   - 平均绝对误差: {results_df['绝对误差'].mean():.3f} 岁")
    print(f"   - 误差标准差: {results_df['绝对误差'].std():.3f} 岁")
    print(f"   - 最大误差: {results_df['绝对误差'].max():.3f} 岁")
    print(f"   - 最小误差: {results_df['绝对误差'].min():.3f} 岁")

if __name__ == "__main__":
    demo_create_results_table() 