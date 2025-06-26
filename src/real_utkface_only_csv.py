#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
100%真实UTKFace数据CSV生成器
只使用真实的UTKFace数据集，不包含任何模拟数据
格式：特征列 | 预测值 | 真实值 | 绝对误差
"""

import os
import sys
import requests
import zipfile
import tarfile
import gdown
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
import cv2
import warnings
warnings.filterwarnings('ignore')

class RealUTKFaceOnlyDownloader:
    """100%真实UTKFace数据下载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def check_existing_data(self) -> bool:
        """检查是否已有真实UTKFace数据"""
        print("🔍 检查现有数据...")
        
        # 搜索所有可能的图像文件位置
        search_patterns = [
            self.data_dir / "*.jpg",
            self.data_dir / "**/*.jpg", 
            self.data_dir / "UTKFace" / "*.jpg",
            self.data_dir / "utkface-new" / "*.jpg",
            self.data_dir / "crop_part1" / "*.jpg",
        ]
        
        all_images = []
        for pattern in search_patterns:
            images = list(Path().glob(str(pattern)))
            all_images.extend(images)
        
        # 验证是否为真实UTKFace格式的文件
        valid_images = []
        for img_path in all_images:
            if self.is_valid_utkface_filename(img_path.name):
                valid_images.append(img_path)
        
        if len(valid_images) > 50:  # 至少需要50个真实图像
            print(f"✅ 发现 {len(valid_images)} 个真实UTKFace图像")
            return True
        else:
            print(f"❌ 只找到 {len(valid_images)} 个有效图像，需要更多真实数据")
            return False
    
    def is_valid_utkface_filename(self, filename: str) -> bool:
        """验证是否为真实UTKFace文件名格式"""
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            
            # UTKFace格式: [age]_[gender]_[race]_[date&time].jpg
            if len(parts) >= 4:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                # 验证范围
                if 0 <= age <= 120 and 0 <= gender <= 1 and 0 <= race <= 4:
                    return True
            return False
        except (ValueError, IndexError):
            return False
    
    def download_utkface_multiple_sources(self) -> bool:
        """从多个源下载真实UTKFace数据"""
        print("🚀 尝试从多个源下载真实UTKFace数据集...")
        
        # 下载源列表
        download_sources = [
            {
                "name": "Google Drive - UTKFace Aligned",
                "method": "gdown",
                "file_id": "0BxYys69jI14kYVM3aVhKS1VhRUk",
                "filename": "UTKFace.tar.gz"
            },
            {
                "name": "Dropbox Mirror",
                "method": "direct",
                "url": "https://www.dropbox.com/s/bg5n8bk8kjxddx5/UTKFace.tar.gz?dl=1",
                "filename": "UTKFace.tar.gz"
            },
            {
                "name": "Archive.org Mirror",
                "method": "direct", 
                "url": "https://archive.org/download/utkface/UTKFace.tar.gz",
                "filename": "UTKFace.tar.gz"
            }
        ]
        
        for source in download_sources:
            print(f"\n🔄 尝试从 {source['name']} 下载...")
            
            try:
                if source['method'] == 'gdown':
                    success = self._download_via_gdown(source['file_id'], source['filename'])
                elif source['method'] == 'direct':
                    success = self._download_direct(source['url'], source['filename'])
                
                if success:
                    print(f"✅ 从 {source['name']} 下载成功!")
                    return True
                    
            except Exception as e:
                print(f"❌ 从 {source['name']} 下载失败: {str(e)}")
                continue
        
        return False
    
    def _download_via_gdown(self, file_id: str, filename: str) -> bool:
        """通过gdown从Google Drive下载"""
        try:
            output_path = self.data_dir / filename
            url = f"https://drive.google.com/uc?id={file_id}"
            
            print(f"   📥 从Google Drive下载: {filename}")
            gdown.download(url, str(output_path), quiet=False)
            
            if output_path.exists() and output_path.stat().st_size > 1024*1024:  # 至少1MB
                return self._extract_archive(output_path)
            return False
            
        except Exception as e:
            print(f"   ❌ gdown下载失败: {str(e)}")
            return False
    
    def _download_direct(self, url: str, filename: str) -> bool:
        """直接HTTP下载"""
        try:
            output_path = self.data_dir / filename
            
            print(f"   📥 直接下载: {filename}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   进度: {progress:.1f}%", end="", flush=True)
            
            print(f"\n   ✅ 下载完成: {output_path}")
            
            if output_path.exists() and output_path.stat().st_size > 1024*1024:  # 至少1MB
                return self._extract_archive(output_path)
            return False
            
        except Exception as e:
            print(f"   ❌ 直接下载失败: {str(e)}")
            return False
    
    def _extract_archive(self, archive_path: Path) -> bool:
        """解压归档文件"""
        print(f"📦 解压文件: {archive_path.name}")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                    
            elif archive_path.suffix in ['.tar', '.gz'] or 'tar.gz' in archive_path.name:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.data_dir)
            
            # 检查解压结果
            jpg_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("**/*.jpg"))
            valid_files = [f for f in jpg_files if self.is_valid_utkface_filename(f.name)]
            
            if len(valid_files) > 50:
                print(f"   ✅ 解压成功: 找到 {len(valid_files)} 个有效UTKFace图像")
                # 删除归档文件以节省空间
                archive_path.unlink()
                return True
            else:
                print(f"   ❌ 解压后只找到 {len(valid_files)} 个有效图像")
                return False
                
        except Exception as e:
            print(f"   ❌ 解压失败: {str(e)}")
            return False
    
    def get_real_data(self) -> bool:
        """获取真实UTKFace数据"""
        print("🎯 获取100%真实UTKFace数据集")
        print("=" * 60)
        
        # 1. 检查现有数据
        if self.check_existing_data():
            return True
        
        # 2. 尝试下载真实数据
        if self.download_utkface_multiple_sources():
            return self.check_existing_data()
        
        # 3. 提供手动下载指导
        print("\n❌ 自动下载失败，请手动下载真实UTKFace数据集:")
        print("📝 手动下载指南:")
        print("   1. 访问官方网站: https://susanqq.github.io/UTKFace/")
        print("   2. 或访问Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("   3. 下载UTKFace数据集")
        print("   4. 解压到当前目录的 data/ 文件夹")
        print("   5. 确保图像文件格式为: [age]_[gender]_[race]_[timestamp].jpg")
        print("   6. 重新运行此脚本")
        print("\n💡 提示: 您也可以将UTKFace图像文件直接放入data/目录")
        
        return False

class RealUTKFaceProcessor:
    """100%真实UTKFace数据处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
    
    def parse_utkface_filename(self, filename: str) -> Optional[dict]:
        """解析UTKFace文件名"""
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            
            if len(parts) >= 4:
                return {
                    'age': int(parts[0]),
                    'gender': int(parts[1]), 
                    'race': int(parts[2]),
                    'timestamp': parts[3] if len(parts) > 3 else '0'
                }
            return None
        except (ValueError, IndexError):
            return None
    
    def extract_real_features(self, image_path: Path) -> Optional[np.ndarray]:
        """从真实图像中提取30维特征"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image)
            img_array = tensor.numpy()
            
            features = []
            
            # RGB通道统计特征 (21维)
            for channel in range(3):
                channel_data = img_array[channel].flatten()
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
                np.sum(all_pixels > np.mean(all_pixels)), # 高于均值的像素数
                np.sum(all_pixels < np.mean(all_pixels)), # 低于均值的像素数
            ]
            features.extend(global_features)
            
            # 纹理特征 (4维)
            gray = np.mean(img_array, axis=0)
            
            # 计算梯度
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
    
    def load_real_dataset(self, max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """加载100%真实UTKFace数据集"""
        print(f"📂 加载真实UTKFace数据集...")
        
        # 搜索所有真实UTKFace图像
        image_patterns = [
            self.data_dir / "*.jpg",
            self.data_dir / "**/*.jpg",
            self.data_dir / "UTKFace" / "*.jpg",
            self.data_dir / "crop_part1" / "*.jpg",
        ]
        
        all_images = []
        for pattern in image_patterns:
            images = list(Path().glob(str(pattern)))
            all_images.extend(images)
        
        # 过滤出有效的UTKFace文件
        valid_images = []
        for img_path in all_images:
            info = self.parse_utkface_filename(img_path.name)
            if info and 0 <= info['age'] <= 120:
                valid_images.append((img_path, info))
        
        if len(valid_images) == 0:
            raise ValueError("❌ 未找到任何有效的真实UTKFace图像文件!")
        
        print(f"   ✅ 找到 {len(valid_images)} 个有效的真实UTKFace图像")
        
        # 限制样本数量
        if len(valid_images) > max_samples:
            valid_images = valid_images[:max_samples]
            print(f"   📊 限制使用前 {max_samples} 个样本")
        
        # 提取特征和标签
        features_list = []
        ages_list = []
        filenames_list = []
        
        print(f"   🔄 正在提取特征...")
        processed = 0
        
        for img_path, info in valid_images:
            features = self.extract_real_features(img_path)
            
            if features is not None:
                features_list.append(features)
                ages_list.append(info['age'])
                filenames_list.append(img_path.name)
                processed += 1
                
                if processed % 50 == 0:
                    print(f"   📊 已处理: {processed}/{len(valid_images)}")
        
        if len(features_list) == 0:
            raise ValueError("❌ 无法从任何图像中提取有效特征!")
        
        print(f"   ✅ 成功处理 {len(features_list)} 个真实样本")
        print(f"   📊 年龄范围: {min(ages_list)}-{max(ages_list)} 岁")
        print(f"   📊 平均年龄: {np.mean(ages_list):.1f} 岁")
        
        return np.array(features_list), np.array(ages_list), filenames_list

def create_real_utkface_csv_only() -> pd.DataFrame:
    """创建100%基于真实UTKFace数据的CSV表格"""
    
    print("🎯 100%真实UTKFace数据CSV表格生成器")
    print("=" * 70)
    print("📋 特点:")
    print("   ✅ 100%使用真实UTKFace图像数据")
    print("   ✅ 不包含任何模拟或生成数据")
    print("   ✅ 直接从真实面部图像提取特征")
    print("   ✅ 使用真实的年龄标签")
    print("=" * 70)
    
    # 1. 下载/检查真实数据
    downloader = RealUTKFaceOnlyDownloader("data")
    if not downloader.get_real_data():
        raise ValueError("无法获取真实UTKFace数据集，请手动下载后重试")
    
    # 2. 加载真实数据集
    processor = RealUTKFaceProcessor("data")
    features, ages, filenames = processor.load_real_dataset(max_samples=1000)
    
    print(f"\n📊 真实数据集统计:")
    print(f"   样本数量: {len(features)}")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   年龄分布:")
    print(f"     18-30岁: {np.sum((ages >= 18) & (ages <= 30))} 个")
    print(f"     31-50岁: {np.sum((ages >= 31) & (ages <= 50))} 个") 
    print(f"     51-70岁: {np.sum((ages >= 51) & (ages <= 70))} 个")
    print(f"     70+岁: {np.sum(ages > 70)} 个")
    
    # 3. 特征标准化
    print(f"\n🔧 数据预处理...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 4. 数据划分
    test_size = min(0.3, 200/len(features))  # 最多30%或200个样本作为测试集
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        features_scaled, ages, filenames, test_size=test_size, random_state=42
    )
    
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 5. 训练年龄预测模型
    print(f"\n🎯 训练年龄预测模型...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    
    # 6. 预测和评估
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    correlation = np.corrcoef(y_test, test_pred)[0,1]
    
    print(f"\n📈 模型性能 (100%真实数据):")
    print(f"   测试集 MAE: {test_mae:.3f} 岁")
    print(f"   测试集 RMSE: {test_rmse:.3f} 岁")
    print(f"   训练集 MAE: {train_mae:.3f} 岁")
    print(f"   相关系数: {correlation:.3f}")
    
    # 7. 创建CSV表格
    print(f"\n📋 创建CSV表格...")
    
    # 生成特征列名
    feature_names = []
    for channel in ['R', 'G', 'B']:
        for stat in ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max']:
            feature_names.append(f'{channel}_{stat}')
    
    for stat in ['global_mean', 'global_std', 'global_var', 'bright_pixels', 'dark_pixels']:
        feature_names.append(stat)
    
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
    print(f"   数据来源: 100%真实UTKFace图像")
    
    return df

def main():
    """主函数"""
    try:
        # 生成基于100%真实数据的CSV表格
        results_df = create_real_utkface_csv_only()
        
        # 保存结果 
        output_dir = Path('results/metrics')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'real_only_utkface_features.csv'
        
        # 重新排列列的顺序：30维特征在前，结果在后
        feature_cols = [col for col in results_df.columns if col not in ['Predicted_Age', 'Actual_Age', 'Abs_Error', 'Filename']]
        result_cols = ['Predicted_Age', 'Actual_Age', 'Abs_Error']
        final_cols = feature_cols + result_cols
        
        final_df = results_df[final_cols].copy()
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n💾 100%真实数据CSV已保存: {output_path}")
        
        # 显示结果预览
        print(f"\n📋 真实数据表格预览:")
        preview_cols = feature_cols[:5] + result_cols
        print(final_df[preview_cols].head().to_string(index=False, float_format='%.3f'))
        
        # 性能统计
        print(f"\n📊 真实数据性能统计:")
        abs_errors = final_df['Abs_Error']
        print(f"   样本总数: {len(final_df)} (100%真实)")
        print(f"   平均绝对误差: {abs_errors.mean():.3f} 岁")
        print(f"   中位数误差: {abs_errors.median():.3f} 岁")
        print(f"   最大误差: {abs_errors.max():.3f} 岁")
        print(f"   最小误差: {abs_errors.min():.3f} 岁")
        
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
        
        print(f"\n🎉 100%真实UTKFace数据CSV表格生成完成！")
        print(f"📁 文件: {output_path}")
        print(f"📋 格式: 30维真实特征 | 预测年龄 | 真实年龄 | 绝对误差")
        print(f"✨ 特点: 完全基于真实UTKFace图像数据，无任何模拟成分")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        print(f"\n💡 解决方案:")
        print(f"   1. 请确保已下载真实UTKFace数据集到 data/ 目录")
        print(f"   2. 图像文件应符合UTKFace格式: [age]_[gender]_[race]_[timestamp].jpg")
        print(f"   3. 至少需要50个有效的真实图像文件")
        print(f"   4. 可从以下网站手动下载:")
        print(f"      - https://susanqq.github.io/UTKFace/")
        print(f"      - https://www.kaggle.com/datasets/jangedoo/utkface-new")
        
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 