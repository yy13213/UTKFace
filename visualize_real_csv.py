#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化100%真实UTKFace数据的CSV表格
绘制预测性能和特征分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_real_csv():
    """加载真实UTKFace数据的CSV文件"""
    csv_path = "results/metrics/manual_real_utkface_features.csv"
    print(f"📊 加载CSV文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ 成功加载 {len(df)} 行数据")
    print(f"📋 数据列数: {df.shape[1]}")
    
    return df

def create_comprehensive_visualization(df):
    """创建综合性可视化图表"""
    print("🎨 创建综合性可视化图表...")
    
    # 创建大图画布
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 预测vs真实年龄散点图
    plt.subplot(3, 4, 1)
    plt.scatter(df['Actual_Age'], df['Predicted_Age'], alpha=0.6, s=20)
    plt.plot([df['Actual_Age'].min(), df['Actual_Age'].max()], 
             [df['Actual_Age'].min(), df['Actual_Age'].max()], 'r--', linewidth=2)
    plt.xlabel('真实年龄')
    plt.ylabel('预测年龄')
    plt.title('预测vs真实年龄对比\n(100%真实UTKFace数据)')
    plt.grid(True, alpha=0.3)
    
    # 计算R²值
    correlation = np.corrcoef(df['Actual_Age'], df['Predicted_Age'])[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 绝对误差分布直方图
    plt.subplot(3, 4, 2)
    plt.hist(df['Abs_Error'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('绝对误差 (年)')
    plt.ylabel('频次')
    plt.title('预测误差分布')
    plt.axvline(df['Abs_Error'].mean(), color='red', linestyle='--', 
                label=f'均值: {df["Abs_Error"].mean():.2f}年')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 年龄分布对比
    plt.subplot(3, 4, 3)
    plt.hist(df['Actual_Age'], bins=20, alpha=0.5, label='真实年龄', color='blue')
    plt.hist(df['Predicted_Age'], bins=20, alpha=0.5, label='预测年龄', color='red')
    plt.xlabel('年龄')
    plt.ylabel('频次')
    plt.title('年龄分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 误差箱线图按年龄段
    plt.subplot(3, 4, 4)
    age_bins = pd.cut(df['Actual_Age'], bins=5, labels=['0-20', '20-40', '40-60', '60-80', '80+'])
    df_with_bins = df.copy()
    df_with_bins['Age_Group'] = age_bins
    sns.boxplot(data=df_with_bins, x='Age_Group', y='Abs_Error')
    plt.title('不同年龄段的预测误差')
    plt.xlabel('年龄段')
    plt.ylabel('绝对误差')
    plt.xticks(rotation=45)
    
    # 5-8. 特征分析（RGB特征的前4个）
    feature_cols = [col for col in df.columns if col not in ['Predicted_Age', 'Actual_Age', 'Abs_Error']]
    
    for i, feature in enumerate(feature_cols[:4]):
        plt.subplot(3, 4, 5+i)
        plt.scatter(df[feature], df['Actual_Age'], alpha=0.5, s=10)
        plt.xlabel(feature)
        plt.ylabel('真实年龄')
        plt.title(f'{feature} vs 年龄')
        plt.grid(True, alpha=0.3)
    
    # 9. 特征相关性热力图（选择前10个特征）
    plt.subplot(3, 4, 9)
    selected_features = feature_cols[:10] + ['Actual_Age']
    corr_matrix = df[selected_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
    plt.title('特征相关性热力图')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 10. 预测性能统计
    plt.subplot(3, 4, 10)
    stats = [
        f"样本数量: {len(df)}",
        f"平均绝对误差: {df['Abs_Error'].mean():.2f}年",
        f"误差标准差: {df['Abs_Error'].std():.2f}年",
        f"最大误差: {df['Abs_Error'].max():.2f}年",
        f"最小误差: {df['Abs_Error'].min():.2f}年",
        f"相关系数: {correlation:.3f}",
        f"R²值: {r_squared:.3f}"
    ]
    
    plt.text(0.1, 0.9, '\n'.join(stats), transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    plt.title('预测性能统计\n(100%真实UTKFace数据)', fontweight='bold')
    
    # 11. 误差累积分布函数
    plt.subplot(3, 4, 11)
    sorted_errors = np.sort(df['Abs_Error'])
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative_prob, linewidth=2)
    plt.xlabel('绝对误差 (年)')
    plt.ylabel('累积概率')
    plt.title('误差累积分布函数')
    plt.grid(True, alpha=0.3)
    
    # 添加关键百分位数
    percentiles = [50, 80, 90, 95]
    for p in percentiles:
        error_p = np.percentile(df['Abs_Error'], p)
        plt.axvline(error_p, color='red', linestyle='--', alpha=0.7)
        plt.text(error_p, p/100, f'{p}%: {error_p:.1f}年', rotation=90)
    
    # 12. 最优和最差预测示例
    plt.subplot(3, 4, 12)
    best_predictions = df.nsmallest(5, 'Abs_Error')
    worst_predictions = df.nlargest(5, 'Abs_Error')
    
    x_pos = np.arange(5)
    width = 0.35
    
    plt.bar(x_pos - width/2, best_predictions['Actual_Age'], width, 
            label='真实年龄', color='green', alpha=0.7)
    plt.bar(x_pos - width/2, best_predictions['Predicted_Age'], width, 
            label='预测年龄', color='lightgreen', alpha=0.7)
    
    plt.bar(x_pos + width/2, worst_predictions['Actual_Age'], width, 
            color='red', alpha=0.7)
    plt.bar(x_pos + width/2, worst_predictions['Predicted_Age'], width, 
            color='lightcoral', alpha=0.7)
    
    plt.xlabel('样本索引')
    plt.ylabel('年龄')
    plt.title('最佳vs最差预测示例')
    plt.legend()
    plt.xticks(x_pos, [f'Best{i+1}' if i < 2.5 else f'Worst{i-4}' for i in range(5)])
    
    plt.tight_layout()
    
    # 保存图表
    output_path = "results/plots/real_utkface_comprehensive_analysis.png"
    Path("results/plots").mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 图表已保存至: {output_path}")
    
    plt.show()

def create_csv_table_visualization(df):
    """创建CSV表格可视化"""
    print("📋 创建CSV表格可视化...")
    
    # 显示前10行数据作为表格
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # 选择要显示的列（前10个特征 + 预测结果）
    display_cols = []
    feature_cols = [col for col in df.columns if col not in ['Predicted_Age', 'Actual_Age', 'Abs_Error']]
    display_cols.extend(feature_cols[:10])  # 前10个特征
    display_cols.extend(['Predicted_Age', 'Actual_Age', 'Abs_Error'])
    
    # 取前15行数据
    display_df = df[display_cols].head(15).round(3)
    
    # 创建表格
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # 设置标题
    plt.title('100%真实UTKFace数据CSV表格预览\n(显示前15行，前10个特征)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 设置列标题样式
    for i in range(len(display_cols)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置预测结果列的颜色
    pred_col_idx = len(display_cols) - 3  # Predicted_Age列的索引
    actual_col_idx = len(display_cols) - 2  # Actual_Age列的索引
    error_col_idx = len(display_cols) - 1   # Abs_Error列的索引
    
    for i in range(1, 16):  # 数据行
        table[(i, pred_col_idx)].set_facecolor('#E3F2FD')
        table[(i, actual_col_idx)].set_facecolor('#E8F5E8')
        table[(i, error_col_idx)].set_facecolor('#FFF3E0')
    
    # 保存表格图
    output_path = "results/plots/real_utkface_csv_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 CSV表格可视化已保存至: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    print("🚀 开始可视化100%真实UTKFace数据的CSV表格")
    print("=" * 60)
    
    try:
        # 加载CSV数据
        df = load_real_csv()
        
        # 数据概览
        print(f"\n📊 数据概览:")
        print(f"   - 总样本数: {len(df)}")
        print(f"   - 特征维度: {df.shape[1] - 3}")  # 减去3个结果列
        print(f"   - 年龄范围: {df['Actual_Age'].min():.0f} - {df['Actual_Age'].max():.0f}岁")
        print(f"   - 平均绝对误差: {df['Abs_Error'].mean():.2f}年")
        print(f"   - 误差标准差: {df['Abs_Error'].std():.2f}年")
        
        # 创建综合分析图表
        create_comprehensive_visualization(df)
        
        # 创建CSV表格可视化
        create_csv_table_visualization(df)
        
        print("\n✅ 所有可视化图表创建完成!")
        print("📁 图表保存位置: results/plots/")
        print("   - real_utkface_comprehensive_analysis.png: 综合分析图表")
        print("   - real_utkface_csv_table.png: CSV表格预览")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 