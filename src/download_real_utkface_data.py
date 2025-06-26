#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实UTKFace数据集下载器
从多个源下载真实的UTKFace数据集
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import requests
import subprocess
from pathlib import Path
import time
import shutil
from typing import Optional, List

class RealUTKFaceDownloader:
    """真实UTKFace数据集下载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # UTKFace数据集的多个下载源
        self.download_sources = [
            {
                "name": "Kaggle UTKFace",
                "url": "https://www.kaggle.com/datasets/jangedoo/utkface-new",
                "method": "kaggle_api"
            },
            {
                "name": "GitHub Mirror 1",
                "url": "https://github.com/aicip/UTKFace/releases/download/v1.0/UTKFace.tar.gz",
                "method": "direct_download",
                "filename": "UTKFace.tar.gz"
            },
            {
                "name": "GitHub Mirror 2", 
                "url": "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk",
                "method": "gdown",
                "filename": "UTKFace.tar.gz"
            },
            {
                "name": "Alternative Mirror",
                "url": "https://susanqq.github.io/UTKFace/",
                "method": "webpage_scraping"
            }
        ]
    
    def check_existing_data(self) -> bool:
        """检查是否已有UTKFace数据"""
        possible_paths = [
            self.data_dir / "UTKFace",
            self.data_dir / "utkface-new",
            self.data_dir,
        ]
        
        for path in possible_paths:
            if path.exists():
                jpg_files = list(path.glob("*.jpg")) + list(path.glob("**/*.jpg"))
                if len(jpg_files) > 100:  # 至少要有100个图像文件
                    print(f"✅ 发现现有UTKFace数据: {path}")
                    print(f"   图像文件数量: {len(jpg_files)}")
                    return True
        return False
    
    def install_requirements(self):
        """安装必要的依赖"""
        print("📦 安装下载依赖...")
        
        try:
            import kaggle
            print("   ✅ kaggle已安装")
        except ImportError:
            print("   📥 安装kaggle...")
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        
        try:
            import gdown
            print("   ✅ gdown已安装")
        except ImportError:
            print("   📥 安装gdown...")
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        
        try:
            import requests
            print("   ✅ requests已安装")
        except ImportError:
            print("   📥 安装requests...")
            subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    
    def download_via_kaggle(self) -> bool:
        """通过Kaggle API下载"""
        print("🔄 尝试通过Kaggle API下载...")
        
        try:
            import kaggle
            
            # 设置Kaggle API凭据路径
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            # 检查API凭据
            credentials_file = kaggle_dir / "kaggle.json"
            if not credentials_file.exists():
                print("❌ 未找到Kaggle API凭据")
                print("   请访问 https://www.kaggle.com/account 下载kaggle.json")
                print("   并放置到 ~/.kaggle/kaggle.json")
                return False
            
            # 下载数据集
            print("   📥 下载jangedoo/utkface-new...")
            kaggle.api.dataset_download_files(
                'jangedoo/utkface-new',
                path=str(self.data_dir),
                unzip=True
            )
            
            # 检查下载结果
            downloaded_files = list(self.data_dir.glob("*.jpg"))
            if len(downloaded_files) > 0:
                print(f"   ✅ 下载成功: {len(downloaded_files)} 个文件")
                return True
            else:
                print("   ❌ 下载的文件中没有找到图像")
                return False
                
        except Exception as e:
            print(f"   ❌ Kaggle下载失败: {str(e)}")
            return False
    
    def download_via_gdown(self, file_id: str, filename: str) -> bool:
        """通过gdown下载Google Drive文件"""
        print("🔄 尝试通过Google Drive下载...")
        
        try:
            import gdown
            
            output_path = self.data_dir / filename
            
            # 从Google Drive下载
            print(f"   📥 下载 {filename}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                str(output_path),
                quiet=False
            )
            
            if output_path.exists():
                print(f"   ✅ 文件下载成功: {output_path}")
                return self.extract_archive(output_path)
            else:
                print("   ❌ 文件下载失败")
                return False
                
        except Exception as e:
            print(f"   ❌ Google Drive下载失败: {str(e)}")
            return False
    
    def download_direct(self, url: str, filename: str) -> bool:
        """直接HTTP下载"""
        print(f"🔄 尝试直接下载: {url}")
        
        try:
            output_path = self.data_dir / filename
            
            print(f"   📥 下载 {filename}...")
            
            # 使用requests下载
            response = requests.get(url, stream=True)
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
            
            print(f"\n   ✅ 文件下载完成: {output_path}")
            return self.extract_archive(output_path)
            
        except Exception as e:
            print(f"\n   ❌ 直接下载失败: {str(e)}")
            return False
    
    def extract_archive(self, archive_path: Path) -> bool:
        """解压归档文件"""
        print(f"📦 解压文件: {archive_path}")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                    
            elif archive_path.suffix in ['.tar', '.gz'] or 'tar.gz' in archive_path.name:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(self.data_dir)
            
            # 检查解压结果
            jpg_files = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("**/*.jpg"))
            
            if len(jpg_files) > 0:
                print(f"   ✅ 解压成功: 找到 {len(jpg_files)} 个图像文件")
                # 删除归档文件以节省空间
                archive_path.unlink()
                return True
            else:
                print("   ❌ 解压后未找到图像文件")
                return False
                
        except Exception as e:
            print(f"   ❌ 解压失败: {str(e)}")
            return False
    
    def download_utkface_sample(self) -> bool:
        """下载UTKFace样本数据集（如果无法获取完整数据集）"""
        print("🔄 尝试下载UTKFace样本数据...")
        
        # 一些已知的UTKFace样本图像URL
        sample_urls = [
            "https://raw.githubusercontent.com/aicip/UTKFace/master/1_0_0_20161219203650636.jpg.chip.jpg",
            "https://raw.githubusercontent.com/aicip/UTKFace/master/2_0_0_20161219203721419.jpg.chip.jpg",
            # 更多样本URL可以在这里添加
        ]
        
        sample_dir = self.data_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        downloaded_count = 0
        
        for i, url in enumerate(sample_urls):
            try:
                filename = f"sample_{i+1}.jpg"
                output_path = sample_dir / filename
                
                print(f"   📥 下载样本 {i+1}...")
                urllib.request.urlretrieve(url, output_path)
                
                if output_path.exists():
                    downloaded_count += 1
                    print(f"   ✅ 样本 {i+1} 下载成功")
                    
            except Exception as e:
                print(f"   ❌ 样本 {i+1} 下载失败: {str(e)}")
        
        if downloaded_count > 0:
            print(f"✅ 成功下载 {downloaded_count} 个样本")
            return True
        else:
            print("❌ 没有成功下载任何样本")
            return False
    
    def download_real_utkface(self) -> bool:
        """下载真实UTKFace数据集"""
        print("🎯 开始下载真实UTKFace数据集")
        print("=" * 50)
        
        # 检查是否已有数据
        if self.check_existing_data():
            print("✅ 已有UTKFace数据，跳过下载")
            return True
        
        # 安装依赖
        self.install_requirements()
        
        # 尝试各种下载方法
        print("\n🔄 尝试下载真实UTKFace数据集...")
        
        # 方法1: Kaggle API
        if self.download_via_kaggle():
            return True
        
        # 方法2: Google Drive
        google_drive_ids = [
            "0BxYys69jI14kYVM3aVhKS1VhRUk",  # 原始UTKFace
            "1BnQGpWPEkHCITg-XzfhwSKWYf7sX7Z3t",  # 备用链接
        ]
        
        for file_id in google_drive_ids:
            if self.download_via_gdown(file_id, "UTKFace.tar.gz"):
                return True
        
        # 方法3: 直接下载
        direct_urls = [
            "https://github.com/aicip/UTKFace/releases/download/v1.0/UTKFace.tar.gz",
            "https://www.dropbox.com/s/bg5n8bk8kjxddx5/UTKFace.tar.gz?dl=1",
        ]
        
        for url in direct_urls:
            if self.download_direct(url, "UTKFace.tar.gz"):
                return True
        
        # 方法4: 下载样本数据
        print("\n⚠️  无法下载完整数据集，尝试下载样本数据...")
        if self.download_utkface_sample():
            print("📝 注意: 只下载了样本数据，不是完整的UTKFace数据集")
            return True
        
        print("\n❌ 所有下载方法都失败了")
        print("💡 建议:")
        print("   1. 手动从 https://susanqq.github.io/UTKFace/ 下载")
        print("   2. 或从 Kaggle 手动下载: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("   3. 解压到 data/ 目录")
        
        return False

def main():
    """主函数"""
    print("🎯 真实UTKFace数据集下载器")
    print("=" * 60)
    
    downloader = RealUTKFaceDownloader("data")
    
    if downloader.download_real_utkface():
        print("\n🎉 UTKFace数据集准备完成!")
        
        # 统计下载的文件
        data_dir = Path("data")
        jpg_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("**/*.jpg"))
        
        print(f"📊 数据集统计:")
        print(f"   图像文件数量: {len(jpg_files)}")
        print(f"   数据目录: {data_dir.absolute()}")
        
        if len(jpg_files) > 0:
            print(f"   示例文件: {jpg_files[0].name}")
        
        print(f"\n💡 接下来可以运行 original_features_csv.py 生成真实数据的CSV表格")
        
    else:
        print("\n❌ UTKFace数据集下载失败")
        print("请手动下载数据集到 data/ 目录")

if __name__ == "__main__":
    main() 