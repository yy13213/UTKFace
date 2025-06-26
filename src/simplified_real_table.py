import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SimplifiedUTKFaceProcessor:
    """简化的UTKFace数据处理器"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = data_path
        
    def parse_filename(self, filename: str) -> Optional[int]:
        """解析UTKFace文件名中的年龄"""
        try:
            basename = os.path.splitext(filename)[0]
            parts = basename.split('_')
            if len(parts) >= 1:
                age = int(parts[0])
                return age if 0 <= age <= 120 else None
            return None
        except (ValueError, IndexError):
            return None
    
    def extract_basic_features(self, image_path: str) -> Optional[np.ndarray]:
        """提取基本图像特征"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            image = image.resize((64, 64))  # 简化为64x64
            
            # 转换为numpy数组
            img_array = np.array(image) / 255.0
            
            # 提取统计特征
            features = []
            
            # RGB通道的统计特征
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75)
                ])
            
            # 灰度统计特征
            gray = np.mean(img_array, axis=2)
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"   错误处理 {image_path}: {str(e)}")
            return None
    
    def load_data(self, max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """加载数据"""
        print(f"🔍 在 {self.data_path} 中搜索UTKFace图像文件...")
        
        # 查找图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            files = glob.glob(os.path.join(self.data_path, ext))
            image_files.extend(files)
        
        if len(image_files) == 0:
            print("❌ 未找到图像文件，生成模拟数据...")
            return self._generate_mock_data(max_samples)
        
        print(f"📸 找到 {len(image_files)} 个图像文件")
        
        # 处理图像文件
        features_list = []
        ages_list = []
        filenames_list = []
        
        processed_count = 0
        
        for img_path in image_files:
            if processed_count >= max_samples:
                break
                
            filename = os.path.basename(img_path)
            age = self.parse_filename(filename)
            
            if age is None:
                continue
                
            features = self.extract_basic_features(img_path)
            if features is None:
                continue
            
            features_list.append(features)
            ages_list.append(age)
            filenames_list.append(filename)
            processed_count += 1
            
            if processed_count % 20 == 0:
                print(f"   已处理 {processed_count} 个样本")
        
        if len(features_list) == 0:
            print("❌ 没有成功处理任何真实数据，生成模拟数据...")
            return self._generate_mock_data(max_samples)
        
        print(f"✅ 成功处理 {len(features_list)} 个真实样本")
        
        return np.array(features_list), np.array(ages_list), filenames_list
    
    def _generate_mock_data(self, max_samples: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """生成模拟数据"""
        np.random.seed(42)
        
        # 生成18维特征
        features = np.random.randn(max_samples, 18)
        
        # 生成年龄（基于特征的合理组合）
        age_base = np.dot(features[:, :5], [2, -1, 1.5, 0.5, -0.8]) + 40
        ages = np.clip(age_base + np.random.normal(0, 8, max_samples), 1, 99).astype(int)
        
        # 生成文件名
        filenames = [f"{age}_{np.random.randint(0,2)}_{np.random.randint(0,5)}_demo_{i:03d}.jpg" 
                    for i, age in enumerate(ages)]
        
        print(f"✅ 生成了 {max_samples} 个模拟样本")
        return features, ages, filenames

def create_prediction_results_csv(data_path: str = "data", 
                                max_samples: int = 100,
                                test_ratio: float = 0.3):
    """创建预测结果CSV文件"""
    
    print("🎯 开始创建UTKFace预测结果表格")
    print("=" * 50)
    
    # 1. 加载数据
    processor = SimplifiedUTKFaceProcessor(data_path)
    features, ages, filenames = processor.load_data(max_samples)
    
    print(f"\n📊 数据概览:")
    print(f"   样本数量: {len(features)}")
    print(f"   特征维度: {features.shape[1]}")
    print(f"   年龄范围: {ages.min()} - {ages.max()} 岁")
    print(f"   平均年龄: {ages.mean():.1f} 岁")
    
    # 2. 数据预处理
    if len(features) >= 10:
        # 分割训练测试集
        X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
            features, ages, filenames, test_size=test_ratio, random_state=42
        )
        print(f"\n📊 数据分割:")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
    else:
        # 样本太少，使用全部数据
        X_train = X_test = features
        y_train = y_test = ages
        files_train = files_test = filenames
        print(f"\n⚠️  样本较少，使用全部数据进行演示")
    
    # 3. 特征降维
    n_components = min(8, X_train.shape[1] - 1, len(X_train) - 1)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"\n🔄 PCA降维: {X_train.shape[1]} -> {n_components}")
    print(f"   累计方差解释比: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 4. 训练预测模型
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
    model.fit(X_train_pca, y_train)
    
    # 训练误差
    train_pred = model.predict(X_train_pca)
    train_mae = mean_absolute_error(y_train, train_pred)
    print(f"\n📈 训练性能: MAE = {train_mae:.2f} 岁")
    
    # 5. 测试集预测
    test_pred = model.predict(X_test_pca)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"📈 测试性能:")
    print(f"   MAE: {test_mae:.2f} 岁")
    print(f"   RMSE: {test_rmse:.2f} 岁")
    
    # 6. 计算误差指标
    abs_errors = np.abs(test_pred - y_test)
    # 修正相对误差计算，避免除以接近0的数
    rel_errors = np.where(y_test > 0, (abs_errors / y_test) * 100, abs_errors * 100)
    rel_errors = np.clip(rel_errors, 0, 1000)  # 限制最大相对误差为1000%
    
    # 7. 创建结果表格
    results_data = {
        '样本编号': range(1, len(y_test) + 1),
        '文件名': files_test,
    }
    
    # 添加PCA特征
    for i in range(n_components):
        results_data[f'PC{i+1}'] = X_test_pca[:, i]
    
    # 添加预测结果
    results_data.update({
        '预测年龄': np.round(test_pred, 1),
        '真实年龄': y_test,
        '绝对误差': np.round(abs_errors, 2),
        '相对误差(%)': np.round(rel_errors, 1),
        '误差等级': ['低' if e <= 3 else '中' if e <= 8 else '高' for e in abs_errors]
    })
    
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 按绝对误差排序
    df = df.sort_values('绝对误差').reset_index(drop=True)
    df['样本编号'] = range(1, len(df) + 1)
    
    return df

def save_comprehensive_results(df: pd.DataFrame, 
                             base_path: str = 'results/metrics/'):
    """保存完整的结果文件"""
    
    os.makedirs(base_path, exist_ok=True)
    
    # 1. 详细结果CSV
    detail_path = os.path.join(base_path, 'utkface_detailed_results.csv')
    df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    
    # 2. 创建汇总统计
    summary_stats = {
        '指标': [
            '总样本数', '平均绝对误差(岁)', '误差标准差(岁)', 
            '中位数误差(岁)', '最小误差(岁)', '最大误差(岁)',
            '平均相对误差(%)', 'RMSE(岁)', '预测平均值', '真实平均值'
        ],
        '数值': [
            len(df),
            f"{df['绝对误差'].mean():.2f}",
            f"{df['绝对误差'].std():.2f}",
            f"{df['绝对误差'].median():.2f}",
            f"{df['绝对误差'].min():.2f}",
            f"{df['绝对误差'].max():.2f}",
            f"{df['相对误差(%)'].mean():.1f}%",
            f"{np.sqrt(np.mean((df['预测年龄'] - df['真实年龄'])**2)):.2f}",
            f"{df['预测年龄'].mean():.1f}",
            f"{df['真实年龄'].mean():.1f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(base_path, 'utkface_summary_stats.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    # 3. 误差分布统计
    error_bins = [0, 2, 5, 10, 20, float('inf')]
    error_labels = ['优秀(0-2岁)', '良好(2-5岁)', '一般(5-10岁)', '较差(10-20岁)', '很差(>20岁)']
    
    distribution_data = []
    for i, label in enumerate(error_labels):
        if i < len(error_bins) - 1:
            mask = (df['绝对误差'] >= error_bins[i]) & (df['绝对误差'] < error_bins[i+1])
        else:
            mask = df['绝对误差'] >= error_bins[i]
        
        count = mask.sum()
        percentage = count / len(df) * 100
        
        distribution_data.append({
            '误差范围': label,
            '样本数': count,
            '占比(%)': f"{percentage:.1f}%"
        })
    
    dist_df = pd.DataFrame(distribution_data)
    dist_path = os.path.join(base_path, 'utkface_error_distribution.csv')
    dist_df.to_csv(dist_path, index=False, encoding='utf-8-sig')
    
    # 4. 输出结果信息
    print(f"\n📁 结果文件已保存:")
    print(f"   📄 详细结果: {detail_path}")
    print(f"   📊 统计汇总: {summary_path}")
    print(f"   📈 误差分布: {dist_path}")
    
    print(f"\n📋 结果预览 (前10行):")
    display_cols = ['样本编号', '文件名', '预测年龄', '真实年龄', '绝对误差', '相对误差(%)', '误差等级']
    print(df[display_cols].head(10).to_string(index=False))
    
    print(f"\n📊 性能概览:")
    print(f"   🎯 平均绝对误差: {df['绝对误差'].mean():.2f} 岁")
    print(f"   📊 误差标准差: {df['绝对误差'].std():.2f} 岁")
    print(f"   🔄 相对误差: {df['相对误差(%)'].mean():.1f}%")
    print(f"   📈 优秀样本 (≤2岁误差): {(df['绝对误差'] <= 2).sum()}/{len(df)} ({(df['绝对误差'] <= 2).mean()*100:.1f}%)")

def main():
    """主函数"""
    print("🎯 UTKFace简化版结果表格生成")
    
    # 检查数据目录
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        print(f"⚠️  创建数据目录: {data_path}")
    
    try:
        # 生成结果表格
        results_df = create_prediction_results_csv(
            data_path=data_path,
            max_samples=150,  # 最多处理150个样本
            test_ratio=0.3    # 30%用于测试
        )
        
        # 保存结果文件
        save_comprehensive_results(results_df)
        
        print(f"\n🎉 处理完成！所有CSV文件已生成。")
        
    except Exception as e:
        print(f"❌ 出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 