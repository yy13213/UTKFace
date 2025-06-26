#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用kagglehub下载UTKFace数据集
"""

import os
import kagglehub
from pathlib import Path
import shutil

def download_utkface_dataset():
    """下载UTKFace数据集"""
    print("🚀 开始下载UTKFace数据集...")
    print("=" * 50)
    
    try:
        # 下载数据集
        print("📥 正在从Kaggle下载数据集...")
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        
        print(f"✅ 数据集下载完成!")
        print(f"📁 下载路径: {path}")
        
        # 检查下载的文件
        download_path = Path(path)
        jpg_files = list(download_path.glob("*.jpg")) + list(download_path.glob("**/*.jpg"))
        
        print(f"📊 发现 {len(jpg_files)} 个图像文件")
        
        # 创建项目data目录
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # 复制文件到项目data目录
        if jpg_files:
            print("📋 正在复制图像文件到项目data目录...")
            copied = 0
            
            for jpg_file in jpg_files:
                dest_file = data_dir / jpg_file.name
                if not dest_file.exists():
                    shutil.copy2(jpg_file, dest_file)
                    copied += 1
                    
                    if copied % 100 == 0:
                        print(f"   已复制: {copied}/{len(jpg_files)}")
            
            print(f"✅ 成功复制 {copied} 个图像文件到 {data_dir.absolute()}")
            
            # 显示一些示例文件
            print(f"\n📋 示例文件:")
            for i, jpg_file in enumerate(list(data_dir.glob("*.jpg"))[:5]):
                print(f"   {jpg_file.name}")
        
        # 验证文件格式
        print(f"\n🔍 验证UTKFace文件格式...")
        valid_files = []
        
        for jpg_file in data_dir.glob("*.jpg"):
            # 检查UTKFace格式: [age]_[gender]_[race]_[timestamp].jpg
            try:
                name_parts = jpg_file.stem.split('_')
                if len(name_parts) >= 4:
                    age = int(name_parts[0])
                    gender = int(name_parts[1])
                    race = int(name_parts[2])
                    
                    if 0 <= age <= 120 and 0 <= gender <= 1 and 0 <= race <= 4:
                        valid_files.append(jpg_file)
            except (ValueError, IndexError):
                continue
        
        print(f"✅ 发现 {len(valid_files)} 个有效的UTKFace格式文件")
        
        if len(valid_files) >= 100:
            print(f"🎉 数据准备完成! 现在可以运行真实数据处理脚本:")
            print(f"   python src/manual_real_utkface_csv.py")
        else:
            print(f"⚠️  有效文件数量不足 (需要至少100个)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print(f"\n💡 可能的解决方案:")
        print(f"   1. 检查网络连接")
        print(f"   2. 验证Kaggle API凭证")
        print(f"   3. 确保有访问该数据集的权限")
        return False

if __name__ == "__main__":
    download_utkface_dataset() 