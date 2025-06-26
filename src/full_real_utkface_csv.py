#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用全部23,705个真实UTKFace数据生成完整CSV表格
100%真实数据，绝无模拟或生成数据
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torchvision.transforms as transforms
import time
from datetime import datetime

class FullRealUTKFaceProcessor:
    """处理全部真实UTKFace数据的完整处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        print(f"🎯 全量真实UTKFace数据处理器初始化")
        print(f"📁 数据目录: {self.data_dir.absolute()}")
    
    def parse_utkface_info(self, filename: str) -> Optional[dict]:
        """解析UTKFace文件名格式: [age]_[gender]_[race]_[timestamp].jpg"""
        try:
            # 移除扩展名
            base_name = Path(filename).stem
            
            # UTKFace格式: age_gender_race_timestamp
            parts = base_name.split('_')
            if len(parts) >= 4:
                age = int(parts[0])
                gender = int(parts[1])  # 0=female, 1=male
                race = int(parts[2])    # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
                timestamp = parts[3]
                
                # 验证数据合理性
                if 0 <= age <= 120 and 0 <= gender <= 1 and 0 <= race <= 4:
                    return {
                        'age': age,
                        'gender': gender,
                        'race': race,
                        'timestamp': timestamp
                    }
            return None
            
        except (ValueError, IndexError) as e:
            return None
    
    def extract_facial_features(self, image_path: Path) -> Optional[np.ndarray]:
        """从真实面部图像中提取30维特征"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            img_array = tensor.numpy()  # shape: (3, 128, 128)
            
            features = []
            
            # RGB通道统计特征 (21维)
            for channel in range(3):
                channel_data = img_array[channel].flatten()
                
                # 7个统计特征
                channel_features = [
                    np.mean(channel_data),      # 均值
                    np.std(channel_data),       # 标准差
                    np.median(channel_data),    # 中位数
                    np.percentile(channel_data, 25),  # 25%分位数
                    np.percentile(channel_data, 75),  # 75%分位数
                    np.min(channel_data),       # 最小值
                    np.max(channel_data),       # 最大值
                ]
                features.extend(channel_features)
            
            # 全局统计特征 (5维)
            all_pixels = img_array.flatten()
            global_features = [
                np.mean(all_pixels),                    # 全局均值
                np.std(all_pixels),                     # 全局标准差
                np.var(all_pixels),                     # 全局方差
                np.sum(all_pixels > np.mean(all_pixels)), # 亮像素数
                np.sum(all_pixels < np.mean(all_pixels)), # 暗像素数
            ]
            features.extend(global_features)
            
            # 纹理特征 (4维) - 基于梯度
            gray = np.mean(img_array, axis=0)  # 转为灰度
            
            # 计算图像梯度
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            
            texture_features = [
                np.mean(np.abs(grad_x)),    # X方向梯度均值
                np.mean(np.abs(grad_y)),    # Y方向梯度均值  
                np.std(grad_x),             # X方向梯度标准差
                np.std(grad_y),             # Y方向梯度标准差
            ]
            features.extend(texture_features)
            
            return np.array(features)
            
        except Exception as e:
            return None
    
    def find_all_utkface_images(self) -> List[Path]:
        """找到所有真实UTKFace图像"""
        print(f"🔍 搜索所有真实UTKFace图像...")
        
        # 多种可能的搜索路径
        search_patterns = [
            self.data_dir / "*.jpg",
            self.data_dir / "*.jpeg",
            self.data_dir / "**/*.jpg", 
            self.data_dir / "**/*.jpeg",
        ]
        
        all_images = set()
        for pattern in search_patterns:
            try:
                images = list(Path().glob(str(pattern)))
                all_images.update(images)
            except:
                continue
        
        print(f"   发现 {len(all_images)} 个图像文件")
        
        # 验证UTKFace格式
        valid_images = []
        invalid_count = 0
        
        for img_path in all_images:
            info = self.parse_utkface_info(img_path.name)
            if info and 0 <= info['age'] <= 120:
                # 检查文件是否存在且可读
                if img_path.exists() and img_path.stat().st_size > 0:
                    valid_images.append(img_path)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        print(f"   ✅ 有效UTKFace文件: {len(valid_images)}")
        print(f"   ❌ 无效文件: {invalid_count}")
        
        return valid_images
    
    def load_full_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[dict]]:
        """加载全部真实UTKFace数据集 - 无样本数量限制"""
        print(f"📂 加载全部真实UTKFace数据集...")
        
        # 找到所有图像
        all_images = self.find_all_utkface_images()
        
        if len(all_images) == 0:
            raise ValueError("❌ 未找到任何有效的真实UTKFace图像文件!")
        
        print(f"   🎯 准备处理 {len(all_images)} 个真实图像 (100%真实数据)")
        
        # 提取特征和标签
        features_list = []
        ages_list = []
        filenames_list = []
        info_list = []
        
        print(f"   🔄 正在从真实图像提取特征...")
        processed = 0
        failed = 0
        start_time = time.time()
        
        # 按批次处理以减少内存占用
        batch_size = 100
        
        for i, img_path in enumerate(all_images):
            # 解析文件信息
            info = self.parse_utkface_info(img_path.name)
            if not info:
                failed += 1
                continue
            
            # 提取特征
            features = self.extract_facial_features(img_path)
            
            if features is not None:
                features_list.append(features)
                ages_list.append(info['age'])
                filenames_list.append(img_path.name)
                info_list.append(info)
                processed += 1
            else:
                failed += 1
            
            # 进度报告
            if (processed + failed) % 500 == 0 or (i + 1) == len(all_images):
                elapsed = time.time() - start_time
                rate = (processed + failed) / elapsed if elapsed > 0 else 0
                eta = (len(all_images) - (processed + failed)) / rate if rate > 0 else 0
                
                print(f"   📊 进度: {processed + failed}/{len(all_images)} "
                      f"(成功: {processed}, 失败: {failed}) "
                      f"[{rate:.1f} 个/秒, ETA: {eta/60:.1f}分钟]")
        
        if len(features_list) == 0:
            raise ValueError("❌ 无法从任何图像中提取有效特征!")
        
        print(f"   ✅ 成功处理 {len(features_list)} 个真实样本")
        print(f"   ⏱️  总耗时: {(time.time() - start_time)/60:.1f} 分钟")
        
        # 统计分析
        ages_array = np.array(ages_list)
        genders = [info['gender'] for info in info_list]
        races = [info['race'] for info in info_list]
        
        print(f"\n📊 完整数据集统计:")
        print(f"   📈 总样本数: {len(features_list)} (100%真实)")
        print(f"   📈 年龄范围: {min(ages_list)}-{max(ages_list)} 岁")
        print(f"   📈 平均年龄: {np.mean(ages_list):.1f} 岁")
        print(f"   📈 年龄分布:")
        print(f"      0-20岁: {np.sum((ages_array >= 0) & (ages_array <= 20))} 个")
        print(f"      21-40岁: {np.sum((ages_array >= 21) & (ages_array <= 40))} 个")
        print(f"      41-60岁: {np.sum((ages_array >= 41) & (ages_array <= 60))} 个")
        print(f"      61+岁: {np.sum(ages_array > 60)} 个")
        print(f"   📈 性别分布: 女性 {genders.count(0)}, 男性 {genders.count(1)}")
        print(f"   📈 种族分布: 白人 {races.count(0)}, 黑人 {races.count(1)}, 亚洲人 {races.count(2)}, 印度人 {races.count(3)}, 其他 {races.count(4)}")
        
        return np.array(features_list), np.array(ages_list), filenames_list, info_list

def create_full_real_utkface_csv() -> str:
    """创建基于全部真实UTKFace数据的完整CSV表格"""
    
    print("🎯 全量真实UTKFace数据CSV表格生成器")
    print("=" * 80)
    print("📋 处理目标:")
    print("   🎯 使用全部23,705个真实UTKFace图像")
    print("   ✅ 100%使用真实数据，绝无模拟数据")
    print("   ✅ 从真实人脸图像提取30维特征")
    print("   ✅ 使用图像文件名中的真实年龄标签")
    print("   📊 生成完整的测试结果CSV表格")
    print("=" * 80)
    
    # 1. 检查数据目录
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir.absolute()}")
        return ""
    
    # 2. 初始化处理器
    processor = FullRealUTKFaceProcessor("data")
    
    # 3. 加载全量真实数据集
    print(f"\n🚀 开始加载全量真实数据集...")
    features, ages, filenames, info_list = processor.load_full_dataset()
    
    print(f"\n📊 全量真实数据集加载完成:")
    print(f"   样本数量: {len(features)} (目标: 23,705)")
    print(f"   特征维度: {features.shape[1]} (30维)")
    print(f"   数据来源: 100%真实UTKFace图像")
    
    # 4. 特征标准化
    print(f"\n🔧 数据预处理...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 5. 数据划分 - 使用更大的测试集
    test_size = 0.3  # 使用30%作为测试集，确保有足够多的测试样本
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        features_scaled, ages, filenames, test_size=test_size, random_state=42, stratify=None
    )
    
    print(f"   训练集: {len(X_train)} 样本 (真实)")
    print(f"   测试集: {len(X_test)} 样本 (真实)")
    
    # 6. 训练年龄预测模型
    print(f"\n🎯 训练年龄预测模型...")
    model = RandomForestRegressor(
        n_estimators=500,      # 增加树的数量
        max_depth=25,          # 增加深度
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"   🔄 正在训练模型...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"   ✅ 模型训练完成 (耗时: {training_time:.1f}秒)")
    
    # 7. 进行预测
    print(f"\n🔮 进行年龄预测...")
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)  # 对训练集也进行预测
    
    # 8. 计算误差
    abs_errors_test = np.abs(y_test - y_pred_test)
    abs_errors_train = np.abs(y_train - y_pred_train)  # 训练集误差
    
    # 9. 创建结果DataFrame - 合并训练集和测试集
    print(f"\n📊 生成完整CSV表格...")
    
    # 特征列名
    feature_names = []
    # RGB特征 (21维)
    for channel in ['R', 'G', 'B']:
        for stat in ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max']:
            feature_names.append(f'{channel}_{stat}')
    
    # 全局特征 (5维)
    global_names = ['global_mean', 'global_std', 'global_var', 'bright_pixels', 'dark_pixels']
    feature_names.extend(global_names)
    
    # 纹理特征 (4维)
    texture_names = ['grad_x_mean', 'grad_y_mean', 'grad_x_std', 'grad_y_std']
    feature_names.extend(texture_names)
    
    # 创建训练集数据框
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['Predicted_Age'] = y_pred_train
    train_df['Actual_Age'] = y_train
    train_df['Abs_Error'] = abs_errors_train
    train_df['Data_Type'] = 'Train'
    
    # 创建测试集数据框
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['Predicted_Age'] = y_pred_test
    test_df['Actual_Age'] = y_test
    test_df['Abs_Error'] = abs_errors_test
    test_df['Data_Type'] = 'Test'
    
    # 合并训练集和测试集
    complete_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 按绝对误差排序
    complete_df = complete_df.sort_values('Abs_Error')
    
    print(f"   ✅ 完整数据框创建完成: {len(complete_df)} 行")
    print(f"      - 训练集样本: {len(train_df)} 个")
    print(f"      - 测试集样本: {len(test_df)} 个")
    
    # 10. 保存完整CSV文件
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"full_real_utkface_complete_{len(complete_df)}samples_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    
    complete_df.to_csv(csv_path, index=False)
    
    # 11. 性能评估 - 分别评估训练集和测试集
    # 测试集性能
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    # 训练集性能
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # 整体性能
    y_all = np.concatenate([y_train, y_test])
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])
    abs_errors_all = np.concatenate([abs_errors_train, abs_errors_test])
    
    mae_all = mean_absolute_error(y_all, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y_all, y_pred_all))
    r2_all = r2_score(y_all, y_pred_all)
    
    print(f"\n📈 完整模型性能评估:")
    print(f"\n🎯 整体性能 (全部{len(complete_df)}个样本):")
    print(f"   平均绝对误差 (MAE): {mae_all:.2f} 年")
    print(f"   均方根误差 (RMSE): {rmse_all:.2f} 年")
    print(f"   R² 决定系数: {r2_all:.3f}")
    print(f"   误差标准差: {np.std(abs_errors_all):.2f} 年")
    print(f"   最小误差: {np.min(abs_errors_all):.2f} 年")
    print(f"   最大误差: {np.max(abs_errors_all):.2f} 年")
    print(f"   中位数误差: {np.median(abs_errors_all):.2f} 年")
    
    print(f"\n📊 训练集性能 ({len(train_df)}个样本):")
    print(f"   平均绝对误差 (MAE): {mae_train:.2f} 年")
    print(f"   均方根误差 (RMSE): {rmse_train:.2f} 年")
    print(f"   R² 决定系数: {r2_train:.3f}")
    
    print(f"\n🔬 测试集性能 ({len(test_df)}个样本):")
    print(f"   平均绝对误差 (MAE): {mae_test:.2f} 年")
    print(f"   均方根误差 (RMSE): {rmse_test:.2f} 年")
    print(f"   R² 决定系数: {r2_test:.3f}")
    
    # 12. 保存详细性能摘要
    summary_data = {
        'Metric': ['Total_Samples', 'Train_Samples', 'Test_Samples', 
                  'Overall_MAE', 'Overall_RMSE', 'Overall_R2',
                  'Train_MAE', 'Train_RMSE', 'Train_R2',
                  'Test_MAE', 'Test_RMSE', 'Test_R2',
                  'Overall_Error_Std', 'Overall_Min_Error', 'Overall_Max_Error', 'Overall_Median_Error'],
        'Value': [len(features), len(X_train), len(X_test),
                 mae_all, rmse_all, r2_all,
                 mae_train, rmse_train, r2_train,
                 mae_test, rmse_test, r2_test,
                 np.std(abs_errors_all), np.min(abs_errors_all), 
                 np.max(abs_errors_all), np.median(abs_errors_all)],
        'Description': ['总样本数', '训练样本数', '测试样本数',
                       '整体平均绝对误差', '整体均方根误差', '整体R²决定系数',
                       '训练集平均绝对误差', '训练集均方根误差', '训练集R²决定系数',
                       '测试集平均绝对误差', '测试集均方根误差', '测试集R²决定系数',
                       '整体误差标准差', '整体最小误差', '整体最大误差', '整体中位数误差']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"full_real_utkface_complete_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n💾 文件保存完成:")
    print(f"   📊 完整结果: {csv_path}")
    print(f"   📋 性能摘要: {summary_path}")
    print(f"   📏 文件大小: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n🎉 全量真实UTKFace完整数据CSV表格生成完成!")
    print(f"   ✅ 使用了 {len(features)} 个真实样本")
    print(f"   ✅ 生成了 {len(complete_df)} 行完整预测结果")
    print(f"   ✅ 包含训练集 {len(train_df)} 个 + 测试集 {len(test_df)} 个样本")
    print(f"   ✅ 格式: 30维特征 + 预测年龄 + 真实年龄 + 绝对误差 + 数据类型")
    print(f"   ✅ 数据真实性: 100%真实UTKFace数据")
    
    return str(csv_path)

def main():
    """主函数"""
    try:
        csv_path = create_full_real_utkface_csv()
        
        if csv_path:
            print(f"\n🌟 任务成功完成!")
            print(f"📁 CSV文件路径: {csv_path}")
        else:
            print(f"\n❌ 任务失败!")
            
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 