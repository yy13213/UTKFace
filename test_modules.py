#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块测试脚本
逐个验证每个模块是否可以正常导入和运行
"""

import sys
import numpy as np
import torch
import traceback

# 添加src目录到路径
sys.path.append('src')

def test_module_import(module_name: str):
    """测试模块导入"""
    try:
        print(f"🧪 测试导入 {module_name}...")
        if module_name == 'dataset':
            import dataset
            print(f"✅ {module_name} 导入成功")
            return True
        elif module_name == 'feature_extractor':
            import feature_extractor
            print(f"✅ {module_name} 导入成功")
            return True
        elif module_name == 'kde_analysis':
            import kde_analysis
            print(f"✅ {module_name} 导入成功")
            return True
        elif module_name == 'correlation_analysis':
            import correlation_analysis
            print(f"✅ {module_name} 导入成功")
            return True
        elif module_name == 'error_prediction':
            import error_prediction
            print(f"✅ {module_name} 导入成功")
            return True
        else:
            print(f"❌ 未知模块: {module_name}")
            return False
    except Exception as e:
        print(f"❌ {module_name} 导入失败: {e}")
        traceback.print_exc()
        return False

def test_dataset_module():
    """测试数据集模块"""
    try:
        print("\n🧪 测试dataset模块功能...")
        from dataset import get_default_transforms
        transforms = get_default_transforms()
        print("✅ 数据预处理变换创建成功")
        return True
    except Exception as e:
        print(f"❌ dataset模块测试失败: {e}")
        return False

def test_feature_extractor_module():
    """测试特征提取模块"""
    try:
        print("\n🧪 测试feature_extractor模块功能...")
        from feature_extractor import FeatureExtractor, AgeRegressor
        
        # 创建特征提取器
        feature_extractor = FeatureExtractor()
        print("✅ 特征提取器创建成功")
        
        # 创建年龄回归器
        age_regressor = AgeRegressor(input_dim=512)
        print("✅ 年龄回归器创建成功")
        
        return True
    except Exception as e:
        print(f"❌ feature_extractor模块测试失败: {e}")
        return False

def test_kde_analysis_module():
    """测试KDE分析模块"""
    try:
        print("\n🧪 测试kde_analysis模块功能...")
        from kde_analysis import PCAReducer, KDECalculator
        
        # 创建PCA降维器
        pca_reducer = PCAReducer(n_components=10)
        print("✅ PCA降维器创建成功")
        
        # 创建KDE计算器
        kde_calculator = KDECalculator()
        print("✅ KDE计算器创建成功")
        
        # 测试小规模数据
        test_features = np.random.randn(100, 512)
        reduced_features = pca_reducer.fit_transform(test_features)
        print(f"✅ PCA降维测试成功: {test_features.shape} → {reduced_features.shape}")
        
        kde_calculator.fit(reduced_features)
        densities = kde_calculator.compute_densities(reduced_features)
        print(f"✅ KDE计算测试成功: 密度范围 {densities.min():.6f}-{densities.max():.6f}")
        
        return True
    except Exception as e:
        print(f"❌ kde_analysis模块测试失败: {e}")
        return False

def test_correlation_analysis_module():
    """测试相关性分析模块"""
    try:
        print("\n🧪 测试correlation_analysis模块功能...")
        from correlation_analysis import KDEMAECorrelationAnalyzer
        
        # 创建分析器
        analyzer = KDEMAECorrelationAnalyzer()
        print("✅ 相关性分析器创建成功")
        
        # 测试小规模数据
        kde_densities = np.random.exponential(0.01, 100)
        mae_values = np.random.exponential(5, 100)
        
        results = analyzer.analyze_correlation(kde_densities, mae_values)
        print(f"✅ 相关性分析测试成功: 相关系数 {results['pearson_correlation']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ correlation_analysis模块测试失败: {e}")
        return False

def test_error_prediction_module():
    """测试误差预测模块"""
    try:
        print("\n🧪 测试error_prediction模块功能...")
        from error_prediction import ErrorPredictor
        
        # 创建预测器
        predictor = ErrorPredictor()
        print("✅ 误差预测器创建成功")
        
        # 测试小规模数据
        kde_densities = np.random.exponential(0.01, 100)
        mae_values = np.random.exponential(5, 100)
        
        results = predictor.fit(kde_densities, mae_values)
        print(f"✅ 误差预测模型训练成功: R² {results['test_r2']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ error_prediction模块测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始模块测试")
    print("=" * 50)
    
    modules = ['dataset', 'feature_extractor', 'kde_analysis', 'correlation_analysis', 'error_prediction']
    
    # 测试模块导入
    import_results = []
    for module in modules:
        success = test_module_import(module)
        import_results.append(success)
    
    # 如果所有模块导入成功，进行功能测试
    if all(import_results):
        print("\n✅ 所有模块导入成功！开始功能测试...")
        
        function_tests = [
            test_dataset_module(),
            test_feature_extractor_module(),
            test_kde_analysis_module(),
            test_correlation_analysis_module(),
            test_error_prediction_module()
        ]
        
        if all(function_tests):
            print("\n🎉 所有测试通过！项目代码准备就绪")
            print("✅ 可以开始运行完整流程")
            return True
        else:
            print("\n❌ 部分功能测试失败")
            return False
    else:
        print("\n❌ 部分模块导入失败")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 