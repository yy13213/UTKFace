#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境依赖检查脚本
验证项目所需的所有Python包是否正确安装
支持虚拟环境检测和管理
"""

import sys
import os
import subprocess
import importlib
from typing import List, Tuple, Dict

def check_virtual_environment() -> Tuple[bool, str]:
    """检查是否在虚拟环境中"""
    # 检查多种虚拟环境指标
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
        os.environ.get('VIRTUAL_ENV') is not None  # 环境变量
    )
    
    if in_venv:
        venv_path = os.environ.get('VIRTUAL_ENV', sys.prefix)
        venv_name = os.path.basename(venv_path)
        return True, f"虚拟环境已激活 ✅ (环境名: {venv_name})"
    else:
        return False, "未检测到虚拟环境 ⚠️ (建议在虚拟环境中运行)"

def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✅"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} ❌ (需要Python 3.7+)"

def check_package_installation(packages: Dict[str, str]) -> List[Tuple[str, bool, str]]:
    """检查包安装情况"""
    results = []
    
    for package, min_version in packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            
            # 特殊处理某些包的版本获取
            if package == 'PIL':
                import PIL
                version = PIL.__version__
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            
            results.append((package, True, f"{version} ✅"))
            
        except ImportError:
            results.append((package, False, f"未安装 ❌"))
        except Exception as e:
            results.append((package, False, f"错误: {str(e)} ❌"))
    
    return results

def install_missing_packages(missing_packages: List[str]) -> bool:
    """尝试安装缺失的包"""
    if not missing_packages:
        return True
    
    print(f"\n🔧 检测到 {len(missing_packages)} 个缺失的包，正在尝试安装...")
    
    try:
        # 构建安装命令
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing_packages
        print(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ 包安装成功！")
            return True
        else:
            print(f"❌ 包安装失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 安装超时")
        return False
    except Exception as e:
        print(f"❌ 安装过程中出现错误: {str(e)}")
        return False

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            return True, f"GPU可用 ✅ (设备数量: {gpu_count}, 主GPU: {gpu_name})"
        else:
            return False, "GPU不可用，将使用CPU ⚠️"
    except ImportError:
        return False, "PyTorch未安装，无法检查GPU ❌"

def generate_venv_setup_script():
    """生成虚拟环境设置脚本"""
    script_content = """@echo off
echo 正在创建UTKFace项目虚拟环境...
echo.

REM 创建虚拟环境
python -m venv utkface_env

REM 激活虚拟环境
call utkface_env\\Scripts\\activate.bat

REM 升级pip
python -m pip install --upgrade pip

REM 安装项目依赖
python -m pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm Pillow

echo.
echo 虚拟环境设置完成！正在验证环境...
python check_environment.py

echo.
echo 提示：下次使用项目时，请先激活虚拟环境：
echo   utkface_env\\Scripts\\activate.bat

pause
"""
    
    with open('setup_venv.bat', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("📝 已生成虚拟环境设置脚本: setup_venv.bat")

def generate_activate_script():
    """生成虚拟环境激活脚本"""
    script_content = """@echo off
echo 正在激活UTKFace项目虚拟环境...
call utkface_env\\Scripts\\activate.bat
echo 虚拟环境已激活！
echo 当前环境：%VIRTUAL_ENV%
echo.
"""
    
    with open('activate_env.bat', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("📝 已生成虚拟环境激活脚本: activate_env.bat")

def main():
    """主函数"""
    print("🔍 UTKFace项目环境检查")
    print("=" * 50)
    
    # 检查虚拟环境
    venv_ok, venv_msg = check_virtual_environment()
    print(f"虚拟环境: {venv_msg}")
    
    # 检查Python版本
    python_ok, python_msg = check_python_version()
    print(f"Python版本: {python_msg}")
    
    if not python_ok:
        print("❌ Python版本不符合要求，请升级到Python 3.7+")
        return False
    
    # 定义需要检查的包
    required_packages = {
        'torch': '1.9.0',
        'torchvision': '0.10.0', 
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'sklearn': '1.0.0',  # scikit-learn导入名为sklearn
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'tqdm': '4.60.0',
        'PIL': '8.3.0'  # Pillow导入名为PIL
    }
    
    print(f"\n📦 检查 {len(required_packages)} 个必需包...")
    print("-" * 50)
    
    # 检查包安装情况
    results = check_package_installation(required_packages)
    missing_packages = []
    
    for package, installed, message in results:
        print(f"{package:12} : {message}")
        if not installed:
            # 映射到实际安装名
            install_name = package
            if package == 'sklearn':
                install_name = 'scikit-learn'
            elif package == 'PIL':
                install_name = 'Pillow'
            missing_packages.append(install_name)
    
    # 检查GPU可用性
    print(f"\n🎮 GPU检查:")
    print("-" * 50)
    gpu_ok, gpu_msg = check_gpu_availability()
    print(f"GPU状态: {gpu_msg}")
    
    # 总结
    print(f"\n📊 检查总结:")
    print("-" * 50)
    installed_count = len([r for r in results if r[1]])
    total_count = len(results)
    
    print(f"已安装包: {installed_count}/{total_count}")
    print(f"缺失包数: {len(missing_packages)}")
    print(f"虚拟环境: {'已激活' if venv_ok else '未激活'}")
    
    if missing_packages:
        print(f"\n❌ 缺失的包: {', '.join(missing_packages)}")
        
        # 询问是否自动安装
        try:
            response = input("\n是否尝试自动安装缺失的包? (y/n): ").lower().strip()
            if response in ['y', 'yes', 'Y']:
                success = install_missing_packages(missing_packages)
                if success:
                    print("\n🔄 重新检查环境...")
                    # 重新检查
                    results = check_package_installation(required_packages)
                    missing_packages = [r[0] for r in results if not r[1]]
            else:
                print("📝 生成安装脚本...")
                if venv_ok:
                    generate_activate_script()
                else:
                    generate_venv_setup_script()
        except KeyboardInterrupt:
            print("\n\n用户取消操作")
            return False
    
    if not missing_packages:
        print("✅ 所有依赖包已正确安装，环境配置完成！")
        
        # 创建环境验证报告
        with open('environment_report.txt', 'w', encoding='utf-8') as f:
            f.write("UTKFace项目环境检查报告\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"虚拟环境: {venv_msg}\n")
            f.write(f"Python版本: {python_msg}\n")
            f.write(f"GPU状态: {gpu_msg}\n\n")
            f.write("包安装情况:\n")
            for package, installed, message in results:
                f.write(f"  {package:12} : {message}\n")
            f.write(f"\n检查时间: {__import__('datetime').datetime.now()}\n")
        
        print("📋 环境报告已保存到: environment_report.txt")
        
        # 如果不在虚拟环境中，给出提示
        if not venv_ok:
            print("\n⚠️  建议：为了更好的项目隔离，建议在虚拟环境中运行项目")
            print("   可运行 setup_venv.bat 来创建并配置虚拟环境")
        
        return True
    else:
        print(f"❌ 仍有 {len(missing_packages)} 个包未安装，请手动安装后重新检查")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 