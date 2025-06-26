#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用代理下载UTKFace数据集
配置VPN代理：7890端口
"""

import os
import kagglehub
from pathlib import Path
import shutil
import requests

def setup_proxy():
    """设置代理配置"""
    proxy_host = "127.0.0.1"
    proxy_port = "7890"
    
    # 设置环境变量代理
    proxy_url = f"http://{proxy_host}:{proxy_port}"
    
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    
    print(f"🔧 已配置代理: {proxy_url}")
    
    # 测试代理连接
    try:
        print("🔍 测试代理连接...")
        response = requests.get("https://httpbin.org/ip", 
                              proxies={
                                  "http": proxy_url,
                                  "https": proxy_url
                              }, 
                              timeout=10)
        if response.status_code == 200:
            ip_info = response.json()
            print(f"✅ 代理连接成功! IP: {ip_info.get('origin', 'unknown')}")
            return True
        else:
            print(f"❌ 代理连接失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 代理连接测试失败: {str(e)}")
        print(f"💡 请确保VPN已启动且监听7890端口")
        return False

def download_utkface_with_proxy():
    """使用代理下载UTKFace数据集"""
    print("🚀 开始通过代理下载UTKFace数据集...")
    print("=" * 60)
    
    # 1. 设置代理
    if not setup_proxy():
        print("❌ 代理配置失败，请检查VPN设置")
        return False
    
    try:
        # 2. 下载数据集
        print("\n📥 正在通过代理从Kaggle下载数据集...")
        print("⏳ 数据集大小约331MB，请耐心等待...")
        
        path = kagglehub.dataset_download("jangedoo/utkface-new")
        
        print(f"✅ 数据集下载完成!")
        print(f"📁 下载路径: {path}")
        
        # 3. 检查下载的文件
        download_path = Path(path)
        jpg_files = list(download_path.glob("*.jpg")) + list(download_path.glob("**/*.jpg"))
        
        print(f"📊 发现 {len(jpg_files)} 个图像文件")
        
        # 4. 创建项目data目录
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # 5. 复制文件到项目data目录
        if jpg_files:
            print("📋 正在复制图像文件到项目data目录...")
            copied = 0
            
            for jpg_file in jpg_files:
                dest_file = data_dir / jpg_file.name
                if not dest_file.exists():
                    shutil.copy2(jpg_file, dest_file)
                    copied += 1
                    
                    if copied % 500 == 0:
                        print(f"   已复制: {copied}/{len(jpg_files)}")
            
            print(f"✅ 成功复制 {copied} 个图像文件到 {data_dir.absolute()}")
            
            # 显示一些示例文件
            print(f"\n📋 示例文件:")
            sample_files = list(data_dir.glob("*.jpg"))[:8]
            for jpg_file in sample_files:
                print(f"   {jpg_file.name}")
        
        # 6. 验证文件格式
        print(f"\n🔍 验证UTKFace文件格式...")
        valid_files = []
        invalid_files = []
        
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
                    else:
                        invalid_files.append(jpg_file)
                else:
                    invalid_files.append(jpg_file)
            except (ValueError, IndexError):
                invalid_files.append(jpg_file)
        
        print(f"✅ 发现 {len(valid_files)} 个有效的UTKFace格式文件")
        if invalid_files:
            print(f"⚠️  发现 {len(invalid_files)} 个格式不符的文件")
        
        # 7. 数据集统计分析
        if valid_files:
            ages = []
            genders = []
            races = []
            
            for jpg_file in valid_files:
                try:
                    name_parts = jpg_file.stem.split('_')
                    ages.append(int(name_parts[0]))
                    genders.append(int(name_parts[1]))
                    races.append(int(name_parts[2]))
                except:
                    continue
            
            print(f"\n📊 数据集统计:")
            print(f"   年龄范围: {min(ages)}-{max(ages)} 岁")
            print(f"   平均年龄: {sum(ages)/len(ages):.1f} 岁")
            print(f"   性别分布: 女性 {genders.count(0)}, 男性 {genders.count(1)}")
            print(f"   种族分布: 白人 {races.count(0)}, 黑人 {races.count(1)}, 亚洲人 {races.count(2)}, 印度人 {races.count(3)}, 其他 {races.count(4)}")
        
        # 8. 检查是否可以开始处理
        if len(valid_files) >= 100:
            print(f"\n🎉 数据准备完成! 真实UTKFace数据集下载成功!")
            print(f"📈 有效样本数: {len(valid_files)}")
            print(f"🚀 现在可以运行真实数据处理脚本:")
            print(f"   python src/manual_real_utkface_csv.py")
            print(f"\n💾 或直接生成CSV表格:")
            print(f"   python -c \"from src.manual_real_utkface_csv import main; main()\"")
        else:
            print(f"⚠️  有效文件数量不足 (需要至少100个，当前{len(valid_files)}个)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print(f"\n💡 可能的解决方案:")
        print(f"   1. 确保VPN已启动并监听7890端口")
        print(f"   2. 检查网络连接")
        print(f"   3. 验证Kaggle API凭证")
        print(f"   4. 确保有访问该数据集的权限")
        
        # 清理代理设置
        for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            if var in os.environ:
                del os.environ[var]
        
        return False

def check_proxy_connection():
    """检查代理连接状态"""
    proxy_url = "http://127.0.0.1:7890"
    
    try:
        response = requests.get("https://httpbin.org/ip", 
                              proxies={
                                  "http": proxy_url,
                                  "https": proxy_url
                              }, 
                              timeout=5)
        if response.status_code == 200:
            return True, response.json().get('origin', 'unknown')
        else:
            return False, f"状态码: {response.status_code}"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print("🔧 UTKFace数据集代理下载器")
    print("=" * 50)
    print("📝 配置信息:")
    print("   代理地址: 127.0.0.1:7890")
    print("   目标数据集: jangedoo/utkface-new")
    print("   预期大小: ~331MB")
    print("=" * 50)
    
    # 预检查代理
    is_connected, info = check_proxy_connection()
    if is_connected:
        print(f"✅ 代理预检查成功，当前IP: {info}")
    else:
        print(f"❌ 代理预检查失败: {info}")
        print("💡 请确保VPN已启动且监听7890端口")
        
        user_input = input("\n是否继续尝试下载? (y/N): ")
        if user_input.lower() != 'y':
            print("👋 下载已取消")
            exit(0)
    
    # 开始下载
    success = download_utkface_with_proxy()
    
    if success:
        print(f"\n🎉 全部完成! UTKFace数据集已准备就绪")
    else:
        print(f"\n❌ 下载失败，请检查配置后重试") 