#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动真实UTKFace数据CSV生成器
需要用户手动下载真实UTKFace数据集，100%使用真实数据
格式：特征列 | 预测值 | 真实值 | 绝对误差
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ManualUTKFaceDataChecker:
    """手动UTKFace数据检查器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def show_download_instructions(self):
        """显示详细下载指导"""
        print("🎯 真实UTKFace数据集手动下载指南")
        print("=" * 70)
        print("📝 由于UTKFace数据集的版权保护，需要您手动下载真实数据：")
        print()
        print("🔗 下载步骤：")
        print("   1. 访问Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("   2. 注册/登录Kaggle账户")
        print("   3. 点击 'Download' 按钮下载数据集")
        print("   4. 解压下载的zip文件")
        print("   5. 将所有.jpg图像文件复制到以下目录：")
        print(f"      {self.data_dir.absolute()}/")
        print()
        print("📋 文件格式要求：")
        print("   - 文件名格式：[age]_[gender]_[race]_[timestamp].jpg")
        print("   - 示例：21_0_1_20170109142408075.jpg (21岁,女性,白人)")
        print("   - 最少需要：100个有效图像文件")
        print()
        print("🔍 替代下载源：")
        print("   - 官方网站: https://susanqq.github.io/UTKFace/")
        print("   - GitHub项目: https://github.com/aicip/UTKFace")
        print()
        print("⚠️  注意事项：")
        print("   - 请确保下载的是完整的UTKFace数据集")
        print("   - 不要修改原始文件名")
        print("   - 图像文件应该是RGB格式的人脸图像")
        print("=" * 70)
    
    def check_real_data(self) -> bool:
        """检查是否有真实UTKFace数据"""
        print("🔍 检查真实UTKFace数据...")
        
        # 搜索所有可能的图像文件
        search_patterns = [
            self.data_dir / "*.jpg",
            self.data_dir / "*.jpeg", 
            self.data_dir / "**/*.jpg",
            self.data_dir / "**/*.jpeg",
            self.data_dir / "UTKFace" / "*.jpg",
            self.data_dir / "utkface-new" / "*.jpg",
            self.data_dir / "crop_part1" / "*.jpg",
        ]
        
        all_images = []
        for pattern in search_patterns:
            try:
                images = list(Path().glob(str(pattern)))
                all_images.extend(images)
            except:
                continue
        
        # 去重
        all_images = list(set(all_images))
        
        # 验证是否为真实UTKFace格式的文件
        valid_images = []
        for img_path in all_images:
            if self.is_valid_utkface_file(img_path):
                valid_images.append(img_path)
        
        print(f"   📊 搜索结果：")
        print(f"   总图像文件: {len(all_images)}")
        print(f"   有效UTKFace文件: {len(valid_images)}")
        
        if len(valid_images) >= 100:
            print(f"   ✅ 找到足够的真实UTKFace数据 ({len(valid_images)} 个文件)")
            
            # 显示一些示例文件
            print(f"   📋 示例文件:")
            for i, img_path in enumerate(valid_images[:5]):
                age = self.parse_age_from_filename(img_path.name)
                print(f"      {img_path.name} (年龄: {age})")
            
            return True
        else:
            print(f"   ❌ 真实数据不足，需要至少100个有效文件")
            return False
    
    def is_valid_utkface_file(self, img_path: Path) -> bool:
        """验证是否为有效的UTKFace文件"""
        try:
            # 检查文件是否存在且为图像
            if not img_path.exists() or img_path.suffix.lower() not in ['.jpg', '.jpeg']:
                return False
            
            # 检查文件名格式
            filename = img_path.stem
            parts = filename.split('_')
            
            # UTKFace格式: [age]_[gender]_[race]_[timestamp]
            if len(parts) >= 4:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                # 验证范围
                if 0 <= age <= 120 and 0 <= gender <= 1 and 0 <= race <= 4:
                    # 尝试加载图像验证
                    try:
                        Image.open(img_path).convert('RGB')
                        return True
                    except:
                        return False
            
            return False
        except (ValueError, IndexError, Exception):
            return False
    
    def parse_age_from_filename(self, filename: str) -> Optional[int]:
        """从文件名解析年龄"""
        try:
            parts = filename.split('_')
            if len(parts) >= 1:
                age = int(parts[0])
                return age if 0 <= age <= 120 else None
            return None
        except (ValueError, IndexError):
            return None

class RealUTKFaceCSVProcessor:
    """真实UTKFace数据CSV处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    def parse_utkface_info(self, filename: str) -> Optional[dict]:
        """解析UTKFace文件信息"""
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            
            if len(parts) >= 4:
                return {
                    'age': int(parts[0]),
                    'gender': int(parts[1]),  # 0: 女性, 1: 男性
                    'race': int(parts[2]),    # 0: 白人, 1: 黑人, 2: 亚洲人, 3: 印度人, 4: 其他
                    'timestamp': parts[3] if len(parts) > 3 else '0'
                }
            return None
        except (ValueError, IndexError):
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
            channel_names = ['R', 'G', 'B']
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
            print(f"   ❌ 特征提取失败 {image_path.name}: {str(e)}")
            return None
    
    def load_real_utkface_dataset(self, max_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """加载真实UTKFace数据集"""
        print(f"📂 加载真实UTKFace数据集...")
        
        # 搜索所有真实UTKFace图像
        search_patterns = [
            self.data_dir / "*.jpg",
            self.data_dir / "*.jpeg",
            self.data_dir / "**/*.jpg", 
            self.data_dir / "**/*.jpeg",
            self.data_dir / "UTKFace" / "*.jpg",
            self.data_dir / "utkface-new" / "*.jpg",
            self.data_dir / "crop_part1" / "*.jpg",
        ]
        
        all_images = []
        for pattern in search_patterns:
            try:
                images = list(Path().glob(str(pattern)))
                all_images.extend(images)
            except:
                continue
        
        # 去重
        all_images = list(set(all_images))
        
        # 过滤出有效的UTKFace文件
        valid_images = []
        for img_path in all_images:
            info = self.parse_utkface_info(img_path.name)
            if info and 0 <= info['age'] <= 120:
                valid_images.append((img_path, info))
        
        if len(valid_images) == 0:
            raise ValueError("❌ 未找到任何有效的真实UTKFace图像文件!")
        
        print(f"   ✅ 找到 {len(valid_images)} 个有效的真实UTKFace图像")
        
        # 限制样本数量并随机打乱
        if len(valid_images) > max_samples:
            import random
            random.shuffle(valid_images)
            valid_images = valid_images[:max_samples]
            print(f"   📊 随机选择 {max_samples} 个样本")
        
        # 提取特征和标签
        features_list = []
        ages_list = []
        filenames_list = []
        
        print(f"   🔄 正在从真实图像提取特征...")
        processed = 0
        failed = 0
        
        for img_path, info in valid_images:
            features = self.extract_facial_features(img_path)
            
            if features is not None:
                features_list.append(features)
                ages_list.append(info['age'])
                filenames_list.append(img_path.name)
                processed += 1
            else:
                failed += 1
                
            if (processed + failed) % 100 == 0:
                print(f"   📊 已处理: {processed + failed}/{len(valid_images)} (成功: {processed}, 失败: {failed})")
        
        if len(features_list) == 0:
            raise ValueError("❌ 无法从任何图像中提取有效特征!")
        
        print(f"   ✅ 成功处理 {len(features_list)} 个真实样本")
        print(f"   📊 年龄范围: {min(ages_list)}-{max(ages_list)} 岁")
        print(f"   📊 平均年龄: {np.mean(ages_list):.1f} 岁")
        
        # 显示年龄分布
        ages_array = np.array(ages_list)
        print(f"   📊 年龄分布:")
        print(f"      0-20岁: {np.sum((ages_array >= 0) & (ages_array <= 20))} 个")
        print(f"      21-40岁: {np.sum((ages_array >= 21) & (ages_array <= 40))} 个")
        print(f"      41-60岁: {np.sum((ages_array >= 41) & (ages_array <= 60))} 个")
        print(f"      61+岁: {np.sum(ages_array > 60)} 个")
        
        return np.array(features_list), np.array(ages_list), filenames_list

def create_manual_real_utkface_csv() -> str:
    """创建基于手动下载真实UTKFace数据的CSV表格"""
    
    print("🎯 手动真实UTKFace数据CSV表格生成器")
    print("=" * 70)
    print("📋 特点:")
    print("   ✅ 100%使用手动下载的真实UTKFace图像")
    print("   ✅ 绝对不使用任何模拟或生成数据")
    print("   ✅ 直接从真实人脸图像提取30维特征")
    print("   ✅ 使用图像文件名中的真实年龄标签")
    print("=" * 70)
    
    # 1. 检查真实数据
    checker = ManualUTKFaceDataChecker("data")
    
    if not checker.check_real_data():
        print("\n❌ 未找到足够的真实UTKFace数据!")
        checker.show_download_instructions()
        print("\n💡 请按照上述指南下载真实数据后重新运行此脚本")
        raise ValueError("需要手动下载真实UTKFace数据集")
    
    # 2. 加载真实数据集
    processor = RealUTKFaceCSVProcessor("data")
    features, ages, filenames = processor.load_real_utkface_dataset(max_samples=2000)
    
    print(f"\n📊 真实数据集统计:")
    print(f"   样本数量: {len(features)} (100%真实)")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   数据来源: 手动下载的真实UTKFace图像")
    
    # 3. 特征标准化
    print(f"\n🔧 数据预处理...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. 数据划分
    test_size = min(0.25, 300/len(features))  # 最多25%或300个样本作为测试集
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        features_scaled, ages, filenames, test_size=test_size, random_state=42
    )
    
    print(f"   训练集: {len(X_train)} 样本 (真实)")
    print(f"   测试集: {len(X_test)} 样本 (真实)")
    
    # 5. 训练年龄预测模型
    print(f"\n🎯 训练年龄预测模型...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6. 预测和评估
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    correlation = np.corrcoef(y_test, test_pred)[0,1]
    
    print(f"\n📈 模型性能 (基于真实UTKFace数据):")
    print(f"   测试集 MAE: {test_mae:.3f} 岁")
    print(f"   测试集 RMSE: {test_rmse:.3f} 岁")
    print(f"   训练集 MAE: {train_mae:.3f} 岁")
    print(f"   相关系数: {correlation:.3f}")
    
    # 7. 创建CSV表格
    print(f"\n📋 创建CSV表格...")
    
    # 生成特征列名（30维）
    feature_names = []
    # RGB通道统计特征 (21维)
    for channel in ['R', 'G', 'B']:
        for stat in ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max']:
            feature_names.append(f'{channel}_{stat}')
    
    # 全局统计特征 (5维)
    for stat in ['global_mean', 'global_std', 'global_var', 'bright_pixels', 'dark_pixels']:
        feature_names.append(stat)
    
    # 纹理特征 (4维)
    for stat in ['grad_x_mean', 'grad_y_mean', 'grad_x_std', 'grad_y_std']:
        feature_names.append(stat)
    
    # 构建表格数据
    table_data = {}
    
    # 添加30维特征
    for i, feature_name in enumerate(feature_names):
        table_data[feature_name] = X_test[:, i]
    
    # 添加预测结果
    abs_errors = np.abs(test_pred - y_test)
    table_data['Predicted_Age'] = np.round(test_pred, 2)
    table_data['Actual_Age'] = y_test
    table_data['Abs_Error'] = np.round(abs_errors, 2)
    table_data['Filename'] = files_test
    
    # 创建DataFrame并按误差排序
    df = pd.DataFrame(table_data)
    df = df.sort_values('Abs_Error').reset_index(drop=True)
    
    print(f"✅ CSV表格创建完成")
    print(f"   行数: {len(df)} (全部为真实测试样本)")
    print(f"   列数: {len(df.columns)}")
    print(f"   数据来源: 100%手动下载的真实UTKFace图像")
    
    # 8. 保存结果
    output_dir = Path('results/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'manual_real_utkface_features.csv'
    
    # 重新排列列的顺序：30维特征在前，结果在后
    feature_cols = feature_names
    result_cols = ['Predicted_Age', 'Actual_Age', 'Abs_Error']
    final_cols = feature_cols + result_cols
    
    final_df = df[final_cols].copy()
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 真实数据CSV已保存: {output_path}")
    
    # 显示结果预览
    print(f"\n📋 真实数据表格预览 (前5行, 前8列):")
    preview_cols = feature_cols[:5] + result_cols
    print(final_df[preview_cols].head().to_string(index=False, float_format='%.3f'))
    
    # 性能统计
    print(f"\n📊 真实数据性能统计:")
    abs_errors = final_df['Abs_Error']
    print(f"   样本总数: {len(final_df)} (100%真实UTKFace)")
    print(f"   平均绝对误差: {abs_errors.mean():.3f} 岁")
    print(f"   中位数误差: {abs_errors.median():.3f} 岁")
    print(f"   最大误差: {abs_errors.max():.3f} 岁")
    print(f"   最小误差: {abs_errors.min():.3f} 岁")
    print(f"   标准差: {abs_errors.std():.3f} 岁")
    
    # 误差分布分析
    excellent = np.sum(abs_errors <= 3)
    good = np.sum((abs_errors > 3) & (abs_errors <= 6))
    fair = np.sum((abs_errors > 6) & (abs_errors <= 10))
    poor = np.sum(abs_errors > 10)
    
    print(f"\n🎯 真实数据误差分布:")
    print(f"   优秀 (≤3岁): {excellent} 个 ({excellent/len(abs_errors)*100:.1f}%)")
    print(f"   良好 (3-6岁): {good} 个 ({good/len(abs_errors)*100:.1f}%)")
    print(f"   一般 (6-10岁): {fair} 个 ({fair/len(abs_errors)*100:.1f}%)")
    print(f"   较差 (>10岁): {poor} 个 ({poor/len(abs_errors)*100:.1f}%)")
    
    print(f"\n🎉 手动真实UTKFace数据CSV表格生成完成！")
    print(f"📁 文件路径: {output_path}")
    print(f"📋 格式: 30维真实特征 | 预测年龄 | 真实年龄 | 绝对误差")
    print(f"✨ 数据保证: 100%基于手动下载的真实UTKFace图像")
    print(f"🚫 不含任何: 模拟数据、生成数据或人工合成数据")
    
    return str(output_path)

def main():
    """主函数"""
    try:
        result_path = create_manual_real_utkface_csv()
        print(f"\n✅ 成功! 真实数据CSV文件已生成: {result_path}")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        
        if "需要手动下载" in str(e):
            print(f"\n📝 解决步骤:")
            print(f"   1. 按照上面的指南下载真实UTKFace数据集")
            print(f"   2. 将所有.jpg文件放入 data/ 目录")
            print(f"   3. 重新运行此脚本")
            print(f"\n🔗 下载链接:")
            print(f"   - Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new")
            print(f"   - 官方: https://susanqq.github.io/UTKFace/")
        else:
            import traceback
            traceback.print_exc()
        
        return None

if __name__ == "__main__":
    main() 