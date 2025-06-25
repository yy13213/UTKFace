#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
误差预测模型
使用KDE密度作为特征，预测MAE误差
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

class ErrorPredictor:
    """基于KDE密度的误差预测器"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names = ['kde_density']
        
    def prepare_features(self, kde_densities: np.ndarray) -> np.ndarray:
        """
        准备特征矩阵
        
        Args:
            kde_densities: KDE密度值
            
        Returns:
            特征矩阵
        """
        # 基础特征：原始KDE密度
        features = kde_densities.reshape(-1, 1)
        
        # 扩展特征：增加非线性变换
        kde_log = np.log(kde_densities + 1e-10).reshape(-1, 1)  # 对数变换
        kde_sqrt = np.sqrt(kde_densities).reshape(-1, 1)  # 平方根变换
        kde_square = (kde_densities ** 2).reshape(-1, 1)  # 平方变换
        kde_inv = (1 / (kde_densities + 1e-10)).reshape(-1, 1)  # 倒数变换
        
        # 组合特征
        features_extended = np.hstack([
            features,      # 原始密度
            kde_log,       # 对数密度
            kde_sqrt,      # 平方根密度
            kde_square,    # 平方密度
            kde_inv        # 倒数密度
        ])
        
        self.feature_names = [
            'kde_density', 'kde_log', 'kde_sqrt', 'kde_square', 'kde_inv'
        ]
        
        return features_extended
    
    def fit(self, kde_densities: np.ndarray, mae_values: np.ndarray, 
            test_size: float = 0.2) -> Dict:
        """
        训练误差预测模型
        
        Args:
            kde_densities: KDE密度值
            mae_values: MAE值 
            test_size: 测试集比例
            
        Returns:
            训练结果字典
        """
        print("🏋️ 开始训练误差预测模型...")
        
        # 准备特征
        X = self.prepare_features(kde_densities)
        y = mae_values
        
        print(f"   特征维度: {X.shape}")
        print(f"   目标变量范围: {y.min():.2f} - {y.max():.2f}")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        self.fitted = True
        
        # 预测
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # 计算指标
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'model_coefficients': self.model.coef_,
            'model_intercept': self.model.intercept_,
            'feature_names': self.feature_names,
            'alpha': self.alpha,
            'train_predictions': y_train_pred,
            'test_predictions': y_test_pred,
            'y_train': y_train,
            'y_test': y_test,
            'X_train': X_train,
            'X_test': X_test
        }
        
        print(f"✅ 模型训练完成:")
        print(f"   训练集 MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
        print(f"   测试集 MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        print(f"   交叉验证 MAE: {cv_mae:.3f} ± {cv_std:.3f}")
        
        return results
    
    def optimize_hyperparameters(self, kde_densities: np.ndarray, mae_values: np.ndarray) -> float:
        """优化超参数"""
        print("🔧 优化Ridge回归超参数...")
        
        # 准备特征
        X = self.prepare_features(kde_densities)
        y = mae_values
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 超参数搜索范围
        alphas = np.logspace(-3, 2, 50)  # 0.001 to 100
        
        # 使用RidgeCV进行交叉验证
        ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
        ridge_cv.fit(X_scaled, y)
        
        best_alpha = ridge_cv.alpha_
        self.alpha = best_alpha
        self.model = Ridge(alpha=best_alpha, random_state=42)
        
        print(f"✅ 最优超参数: alpha = {best_alpha:.4f}")
        
        return best_alpha
    
    def predict(self, kde_densities: np.ndarray) -> np.ndarray:
        """预测MAE值"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        X = self.prepare_features(kde_densities)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> Dict:
        """获取特征重要性"""
        if not self.fitted:
            return {}
        
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            importance_dict[name] = abs(self.model.coef_[i])
        
        # 按重要性排序
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, save_path: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'alpha': self.alpha,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"📁 模型已保存到: {save_path}")
    
    def load_model(self, save_path: str):
        """加载模型"""
        with open(save_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.alpha = model_data['alpha']
        self.feature_names = model_data['feature_names']
        self.fitted = model_data['fitted']
        
        print(f"📁 模型已从 {save_path} 加载")

def plot_prediction_results(results: Dict, save_path: Optional[str] = None):
    """绘制预测结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('误差预测模型结果', fontsize=16, fontweight='bold')
    
    # 训练集预测vs真实
    axes[0, 0].scatter(results['y_train'], results['train_predictions'], alpha=0.6, s=2)
    min_val = min(results['y_train'].min(), results['train_predictions'].min())
    max_val = max(results['y_train'].max(), results['train_predictions'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('真实MAE')
    axes[0, 0].set_ylabel('预测MAE')
    axes[0, 0].set_title(f'训练集 (R²={results["train_r2"]:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 测试集预测vs真实
    axes[0, 1].scatter(results['y_test'], results['test_predictions'], alpha=0.6, s=2)
    min_val = min(results['y_test'].min(), results['test_predictions'].min())
    max_val = max(results['y_test'].max(), results['test_predictions'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('真实MAE')
    axes[0, 1].set_ylabel('预测MAE')
    axes[0, 1].set_title(f'测试集 (R²={results["test_r2"]:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差分析
    train_residuals = results['y_train'] - results['train_predictions']
    test_residuals = results['y_test'] - results['test_predictions']
    
    axes[0, 2].scatter(results['train_predictions'], train_residuals, alpha=0.5, s=1, label='训练集')
    axes[0, 2].scatter(results['test_predictions'], test_residuals, alpha=0.5, s=1, label='测试集')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 2].set_xlabel('预测MAE')
    axes[0, 2].set_ylabel('残差')
    axes[0, 2].set_title('残差分析')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 模型系数
    if 'model_coefficients' in results and 'feature_names' in results:
        coefs = results['model_coefficients']
        feature_names = results['feature_names']
        
        bars = axes[1, 0].barh(feature_names, coefs)
        axes[1, 0].set_xlabel('系数值')
        axes[1, 0].set_title('模型特征系数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, coef in zip(bars, coefs):
            axes[1, 0].text(coef + 0.01 if coef >= 0 else coef - 0.01, 
                           bar.get_y() + bar.get_height()/2,
                           f'{coef:.3f}', ha='left' if coef >= 0 else 'right', va='center')
    
    # 误差分布对比
    axes[1, 1].hist(train_residuals, bins=30, alpha=0.5, label='训练集残差', density=True)
    axes[1, 1].hist(test_residuals, bins=30, alpha=0.5, label='测试集残差', density=True)
    axes[1, 1].set_xlabel('残差值')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('残差分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 性能指标对比
    metrics_names = ['MAE', 'RMSE', 'R²']
    train_metrics = [results['train_mae'], results['train_rmse'], results['train_r2']]
    test_metrics = [results['test_mae'], results['test_rmse'], results['test_r2']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, train_metrics, width, label='训练集', alpha=0.7)
    axes[1, 2].bar(x + width/2, test_metrics, width, label='测试集', alpha=0.7)
    axes[1, 2].set_xlabel('指标')
    axes[1, 2].set_ylabel('值')
    axes[1, 2].set_title('性能指标对比')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics_names)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
        axes[1, 2].text(i - width/2, train_val + 0.01, f'{train_val:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        axes[1, 2].text(i + width/2, test_val + 0.01, f'{test_val:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 预测结果图已保存到: {save_path}")
    
    plt.show()

def run_error_prediction(kde_densities: np.ndarray, mae_values: np.ndarray) -> Dict:
    """
    运行完整的误差预测流程
    
    Args:
        kde_densities: KDE密度值
        mae_values: MAE值
        
    Returns:
        预测结果字典
    """
    print(f"🚀 开始任务7：误差预测模型")
    print(f"   样本数量: {len(kde_densities)}")
    
    # 创建预测器
    predictor = ErrorPredictor()
    
    # 优化超参数
    best_alpha = predictor.optimize_hyperparameters(kde_densities, mae_values)
    
    # 训练模型
    results = predictor.fit(kde_densities, mae_values)
    
    # 绘制结果
    plot_prediction_results(results, 'results/plots/error_prediction_results.png')
    
    # 获取特征重要性
    feature_importance = predictor.get_feature_importance()
    print(f"✅ 特征重要性:")
    for feature, importance in feature_importance.items():
        print(f"   {feature}: {importance:.4f}")
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    predictor.save_model('models/error_predictor.pkl')
    
    # 保存结果
    prediction_results = {
        'kde_density': kde_densities,
        'true_mae': mae_values,
        'predicted_mae_train': results['train_predictions'] if len(results['train_predictions']) > 0 else [],
        'predicted_mae_test': results['test_predictions'] if len(results['test_predictions']) > 0 else []
    }
    
    # 为所有数据预测
    all_predictions = predictor.predict(kde_densities)
    prediction_results['predicted_mae_all'] = all_predictions
    
    results_df = pd.DataFrame(prediction_results)
    results_df.to_csv('results/error_prediction_results.csv', index=False)
    
    # 生成模型报告
    report = generate_prediction_report(results, feature_importance)
    with open('results/error_prediction_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 任务7完成！")
    print(f"   测试集R²: {results['test_r2']:.3f}")
    print(f"   测试集MAE: {results['test_mae']:.3f}")
    print(f"   最优正则化参数: {best_alpha:.4f}")
    print(f"   结果已保存到: results/")
    
    return results

def generate_prediction_report(results: Dict, feature_importance: Dict) -> str:
    """生成预测模型报告"""
    
    # 判断模型性能
    test_r2 = results['test_r2']
    if test_r2 > 0.5:
        performance = "优秀"
    elif test_r2 > 0.3:
        performance = "良好"
    elif test_r2 > 0.1:
        performance = "一般"
    else:
        performance = "较差"
    
    report = f"""
# 误差预测模型报告

## 模型概述
- **模型类型**: Ridge回归
- **特征**: KDE密度及其变换
- **目标**: 预测MAE误差
- **性能评价**: {performance}

## 性能指标

### 训练集表现
- MAE: {results['train_mae']:.3f}
- RMSE: {results['train_rmse']:.3f}
- R²: {results['train_r2']:.3f}

### 测试集表现  
- MAE: {results['test_mae']:.3f}
- RMSE: {results['test_rmse']:.3f}
- R²: {results['test_r2']:.3f}

### 交叉验证结果
- 平均MAE: {results['cv_mae']:.3f} ± {results['cv_std']:.3f}

## 模型配置
- 正则化参数α: {results['alpha']:.4f}
- 特征数量: {len(results['feature_names'])}
- 截距: {results['model_intercept']:.4f}

## 特征重要性分析
"""
    
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        report += f"{i}. **{feature}**: {importance:.4f}\n"
    
    report += f"""

## 模型解释

### 性能分析
{'模型能够较好地预测误差，具有实用价值。' if test_r2 > 0.3 else '模型预测能力有限，可能需要更多特征或不同方法。'}

R²为{test_r2:.3f}，表明模型能够解释{test_r2*100:.1f}%的误差变异。

### 特征分析
最重要的特征是{list(feature_importance.keys())[0]}，说明{'原始KDE密度' if list(feature_importance.keys())[0] == 'kde_density' else 'KDE密度的变换形式'}对误差预测最有用。

## 应用建议

1. **预测置信度**: 当R²>{test_r2:.1f}时，可以较为可靠地使用该模型预测误差
2. **适用范围**: 模型适用于KDE密度在[{results.get('kde_min', 'N/A'):.6f}, {results.get('kde_max', 'N/A'):.6f}]范围内的样本
3. **改进方向**: {'考虑增加更多特征或使用非线性模型' if test_r2 < 0.5 else '当前模型性能良好，可直接应用'}

## 技术细节
- 特征标准化: StandardScaler
- 正则化: L2 Ridge回归  
- 交叉验证: 5折
- 超参数优化: RidgeCV网格搜索
"""
    
    return report

if __name__ == "__main__":
    # 测试代码
    print("🧪 误差预测模型测试...")
    
    # 生成测试数据
    np.random.seed(42)
    kde_densities = np.random.exponential(0.01, 1000)
    # 创建与KDE有关的MAE（模拟负相关关系）
    mae_values = 5 + 2 * np.log(1/(kde_densities + 1e-10)) + np.random.normal(0, 1, 1000)
    mae_values = np.abs(mae_values)  # 确保MAE为正
    
    print("📝 使用模拟数据进行测试")
    results = run_error_prediction(kde_densities, mae_values)
    
    print("✅ 测试完成！")
    print("📝 实际使用时请传入真实的KDE密度和MAE数据") 