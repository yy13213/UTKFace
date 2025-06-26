import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from typing import Tuple, List, Optional
import re
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class UTKFaceRealDataProcessor:
    """UTKFace真实数据处理器"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def parse_filename(self, filename: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        解析UTKFace文件名
        格式: [age]_[gender]_[race]_[date&time].jpg
        
        Returns:
            age, gender, race (如果解析失败返回None)
        """
        try:
            # 移除文件扩展名
            basename = os.path.splitext(filename)[0]
            parts = basename.split('_')
            
            if len(parts) >= 3:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                return age, gender, race
            else:
                return None, None, None
        except (ValueError, IndexError):
            return None, None, None
    
    def load_sample_data(self, max_samples: int = 200) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """
        加载样本数据（如果没有真实数据，生成模拟数据）
        
        Args:
            max_samples: 最大样本数
            
        Returns:
            features: 特征数组 (n_samples, n_features)
            ages: 年龄数组 (n_samples,)
            filenames: 文件名列表
            sample_ids: 样本ID列表
        """
        # 尝试加载真实数据
        if os.path.exists(self.data_path):
            image_files = glob.glob(os.path.join(self.data_path, "*.jpg"))
            if len(image_files) > 0:
                print(f"🎯 发现 {len(image_files)} 个真实图像文件")
                return self._load_real_data(image_files, max_samples)
        
        # 如果没有真实数据，生成模拟数据
        print("⚠️  未找到真实UTKFace数据，生成模拟数据进行演示...")
        return self._generate_simulation_data(max_samples)
    
    def _load_real_data(self, image_files: List[str], max_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """加载真实图像数据"""
        features_list = []
        ages_list = []
        filenames_list = []
        sample_ids = []
        
        # 限制处理的文件数量
        if len(image_files) > max_samples:
            image_files = image_files[:max_samples]
        
        print(f"📸 正在处理 {len(image_files)} 个图像文件...")
        
        for i, img_path in enumerate(image_files):
            try:
                # 解析文件名获取年龄
                filename = os.path.basename(img_path)
                age, gender, race = self.parse_filename(filename)
                
                if age is None:
                    continue
                
                # 加载和预处理图像
                image = Image.open(img_path).convert('RGB')
                tensor = self.transform(image)
                
                # 提取简单的图像特征（均值、方差等）
                features = self._extract_simple_features(tensor)
                
                features_list.append(features)
                ages_list.append(age)
                filenames_list.append(filename)
                sample_ids.append(i)
                
                if len(features_list) % 50 == 0:
                    print(f"   已处理 {len(features_list)} 个样本...")
                    
            except Exception as e:
                print(f"   跳过文件 {filename}: {str(e)}")
                continue
        
        if len(features_list) == 0:
            print("❌ 没有成功加载任何真实数据，转为生成模拟数据")
            return self._generate_simulation_data(max_samples)
        
        features = np.array(features_list)
        ages = np.array(ages_list)
        
        print(f"✅ 成功加载 {len(features)} 个真实样本")
        
        return features, ages, filenames_list, sample_ids
    
    def _extract_simple_features(self, tensor: torch.Tensor) -> np.ndarray:
        """提取简单的图像特征"""
        # 转换为numpy数组
        img_array = tensor.numpy()
        
        features = []
        
        # 对每个通道计算统计特征
        for channel in range(img_array.shape[0]):
            channel_data = img_array[channel]
            
            # 基本统计特征
            features.extend([
                np.mean(channel_data),      # 均值
                np.std(channel_data),       # 标准差
                np.min(channel_data),       # 最小值
                np.max(channel_data),       # 最大值
                np.median(channel_data),    # 中位数
            ])
        
        # 全局特征
        all_pixels = img_array.flatten()
        features.extend([
            np.mean(all_pixels),
            np.std(all_pixels),
            np.percentile(all_pixels, 25),  # 25%分位数
            np.percentile(all_pixels, 75),  # 75%分位数
            np.sum(all_pixels > 0),         # 非零像素数
        ])
        
        return np.array(features)
    
    def _generate_simulation_data(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        """生成模拟数据"""
        np.random.seed(42)
        
        # 生成模拟特征（20维）
        features = np.random.randn(max_samples, 20)
        
        # 生成模拟年龄（基于特征的线性组合加噪声）
        age_coeffs = np.random.randn(20) * 0.5
        base_ages = np.dot(features, age_coeffs) * 10 + 40  # 中心年龄40岁
        ages = np.clip(base_ages + np.random.normal(0, 5, max_samples), 0, 100).astype(int)
        
        # 生成模拟文件名
        filenames = []
        for i in range(max_samples):
            gender = np.random.randint(0, 2)
            race = np.random.randint(0, 5)
            timestamp = f"20200101_{i:04d}"
            filename = f"{ages[i]}_{gender}_{race}_{timestamp}.jpg"
            filenames.append(filename)
        
        sample_ids = list(range(max_samples))
        
        print(f"✅ 生成了 {max_samples} 个模拟样本")
        
        return features, ages, filenames, sample_ids

class AgePredictor:
    """年龄预测器"""
    
    def __init__(self, use_pca: bool = True, n_components: int = 10):
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) if use_pca else None
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_fitted = False
    
    def fit(self, features: np.ndarray, ages: np.ndarray):
        """训练预测模型"""
        print("🎯 训练年龄预测模型...")
        
        # PCA降维
        if self.use_pca:
            features_transformed = self.pca.fit_transform(features)
            print(f"   PCA降维: {features.shape[1]} -> {self.n_components}")
        else:
            features_transformed = features
        
        # 训练回归模型
        self.regressor.fit(features_transformed, ages)
        self.is_fitted = True
        
        # 计算训练误差
        train_pred = self.regressor.predict(features_transformed)
        train_mae = mean_absolute_error(ages, train_pred)
        train_rmse = np.sqrt(mean_squared_error(ages, train_pred))
        
        print(f"   训练MAE: {train_mae:.3f} 岁")
        print(f"   训练RMSE: {train_rmse:.3f} 岁")
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测年龄"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # PCA变换
        if self.use_pca:
            features_transformed = self.pca.transform(features)
        else:
            features_transformed = features
        
        # 预测
        predictions = self.regressor.predict(features_transformed)
        return predictions
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        if self.use_pca:
            return [f'PC{i+1}' for i in range(self.n_components)]
        else:
            return [f'特征_{i+1}' for i in range(self.regressor.n_features_in_)]

def create_real_data_results_table(data_path: str = "data", 
                                 max_samples: int = 150,
                                 test_size: float = 0.3) -> pd.DataFrame:
    """
    使用真实数据创建结果表格
    
    Args:
        data_path: 数据路径
        max_samples: 最大样本数
        test_size: 测试集比例
        
    Returns:
        pd.DataFrame: 结果表格
    """
    print("🚀 开始处理真实UTKFace数据...")
    
    # 1. 加载数据
    processor = UTKFaceRealDataProcessor(data_path)
    features, ages, filenames, sample_ids = processor.load_sample_data(max_samples)
    
    print(f"\n📊 数据集信息:")
    print(f"   样本数量: {len(features)}")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   年龄范围: {ages.min()} - {ages.max()} 岁")
    print(f"   平均年龄: {ages.mean():.1f} 岁")
    
    # 2. 划分训练测试集
    if len(features) > 10:  # 至少需要10个样本才能划分
        X_train, X_test, y_train, y_test, idx_train, idx_test, files_train, files_test = train_test_split(
            features, ages, sample_ids, filenames, test_size=test_size, random_state=42
        )
        print(f"\n📊 数据划分:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
    else:
        # 样本太少，全部用作测试
        X_train, X_test = features, features
        y_train, y_test = ages, ages
        idx_train, idx_test = sample_ids, sample_ids
        files_train, files_test = filenames, filenames
        print(f"\n⚠️  样本数量较少，使用全部数据")
    
    # 3. 训练预测模型
    predictor = AgePredictor(use_pca=True, n_components=min(10, features.shape[1]))
    predictor.fit(X_train, y_train)
    
    # 4. 在测试集上预测
    print(f"\n🎯 在测试集上进行预测...")
    predictions = predictor.predict(X_test)
    
    # 5. 计算误差
    abs_errors = np.abs(predictions - y_test)
    rel_errors = (abs_errors / np.maximum(y_test, 1e-6)) * 100
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"📈 测试集性能:")
    print(f"   MAE: {mae:.3f} 岁")
    print(f"   RMSE: {rmse:.3f} 岁")
    print(f"   平均相对误差: {rel_errors.mean():.1f}%")
    
    # 6. 获取特征值（测试集的PCA变换结果）
    if predictor.use_pca:
        test_features_transformed = predictor.pca.transform(X_test)
    else:
        test_features_transformed = X_test
    
    # 7. 创建结果表格
    print(f"\n📋 创建结果表格...")
    
    # 构建表格数据
    table_data = {
        '样本ID': idx_test,
        '文件名': files_test,
    }
    
    # 添加特征列
    feature_names = predictor.get_feature_names()
    for i, feature_name in enumerate(feature_names):
        table_data[feature_name] = test_features_transformed[:, i]
    
    # 添加预测结果列
    table_data['预测值'] = predictions
    table_data['真实值'] = y_test
    table_data['绝对误差'] = abs_errors
    table_data['相对误差(%)'] = rel_errors
    
    # 创建DataFrame
    results_df = pd.DataFrame(table_data)
    
    # 按绝对误差排序，展示不同误差水平的样本
    results_df = results_df.sort_values('绝对误差').reset_index(drop=True)
    
    return results_df

def save_results_to_csv(df: pd.DataFrame, 
                       csv_path: str = 'results/metrics/utkface_real_results.csv',
                       summary_path: str = 'results/metrics/utkface_summary.csv'):
    """
    保存结果到CSV文件
    
    Args:
        df: 结果DataFrame
        csv_path: 详细结果CSV路径
        summary_path: 统计摘要CSV路径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 1. 保存详细结果
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 详细结果已保存到: {csv_path}")
    
    # 2. 创建统计摘要
    abs_errors = df['绝对误差'].values
    
    # 误差区间统计
    error_bins = [0, 1, 2, 3, 5, 10, float('inf')]
    error_labels = ['0-1岁', '1-2岁', '2-3岁', '3-5岁', '5-10岁', '>10岁']
    
    error_stats = []
    cumulative_percentage = 0
    
    for i in range(len(error_bins)-1):
        mask = (abs_errors >= error_bins[i]) & (abs_errors < error_bins[i+1])
        count = np.sum(mask)
        percentage = count / len(abs_errors) * 100
        cumulative_percentage += percentage
        
        error_stats.append({
            '误差区间': error_labels[i],
            '样本数量': count,
            '占比(%)': f'{percentage:.1f}%',
            '累计占比(%)': f'{cumulative_percentage:.1f}%'
        })
    
    # 详细统计信息
    detailed_stats = {
        '统计指标': [
            '样本总数', '平均绝对误差', '误差标准差', '中位数误差',
            '最小误差', '最大误差', '25%分位数', '75%分位数',
            '平均相对误差(%)', 'RMSE', '预测值均值', '真实值均值'
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
            f'{np.sqrt(np.mean((df["预测值"] - df["真实值"])**2)):.3f}',
            f'{df["预测值"].mean():.1f}',
            f'{df["真实值"].mean():.1f}'
        ]
    }
    
    # 合并统计数据
    summary_data = []
    
    # 添加误差分布统计
    for stat in error_stats:
        summary_data.append(stat)
    
    # 添加分隔行
    summary_data.append({'误差区间': '---详细统计---', '样本数量': '', '占比(%)': '', '累计占比(%)': ''})
    
    # 添加详细统计
    for i, metric in enumerate(detailed_stats['统计指标']):
        summary_data.append({
            '误差区间': metric,
            '样本数量': detailed_stats['数值'][i],
            '占比(%)': '',
            '累计占比(%)': ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"✅ 统计摘要已保存到: {summary_path}")
    
    # 3. 显示结果预览
    print(f"\n📋 结果表格预览 (前10行):")
    display_columns = ['样本ID', '文件名', '预测值', '真实值', '绝对误差', '相对误差(%)']
    if 'PC1' in df.columns:
        display_columns.insert(2, 'PC1')
        display_columns.insert(3, 'PC2')
    
    print(df[display_columns].head(10).to_string(index=False))
    
    print(f"\n📊 数据概览:")
    print(f"   总样本数: {len(df)}")
    print(f"   平均绝对误差: {df['绝对误差'].mean():.3f} 岁")
    print(f"   误差标准差: {df['绝对误差'].std():.3f} 岁")
    print(f"   最大误差: {df['绝对误差'].max():.3f} 岁")
    print(f"   最小误差: {df['绝对误差'].min():.3f} 岁")

def main():
    """主函数"""
    print("🎯 UTKFace真实数据结果表格生成")
    print("=" * 50)
    
    # 检查数据目录
    data_path = "data"
    if not os.path.exists(data_path):
        print(f"⚠️  数据目录 {data_path} 不存在，将创建并使用模拟数据")
        os.makedirs(data_path, exist_ok=True)
    
    try:
        # 创建结果表格
        results_df = create_real_data_results_table(
            data_path=data_path,
            max_samples=200,  # 处理最多200个样本
            test_size=0.3     # 30%作为测试集
        )
        
        # 保存到CSV
        save_results_to_csv(
            df=results_df,
            csv_path='results/metrics/utkface_real_results.csv',
            summary_path='results/metrics/utkface_summary.csv'
        )
        
        print(f"\n🎉 处理完成！")
        print(f"📁 结果文件:")
        print(f"   - 详细结果: results/metrics/utkface_real_results.csv")
        print(f"   - 统计摘要: results/metrics/utkface_summary.csv")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 