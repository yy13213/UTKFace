#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTKFace演示数据生成器
创建少量符合UTKFace格式的示例图像，用于演示真实数据处理流程
仅用于演示目的，不代表真实的UTKFace数据集
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random

def create_demo_utkface_images(data_dir: str = "data", num_samples: int = 150):
    """创建演示用的UTKFace格式图像"""
    
    print("🎭 创建UTKFace格式演示图像...")
    print("⚠️  注意：这些是演示用的人工图像，不是真实的UTKFace数据集！")
    print("📝 真实数据请从官方渠道下载")
    print("=" * 60)
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # 清理已有的演示文件
    for file in data_path.glob("demo_*.jpg"):
        file.unlink()
    
    created_files = []
    
    for i in range(num_samples):
        # 随机生成UTKFace格式的信息
        age = random.randint(18, 80)
        gender = random.randint(0, 1)  # 0: 女性, 1: 男性
        race = random.randint(0, 4)    # 0-4: 不同种族
        timestamp = f"2017010{i%9}14240{i%9}075"
        
        # 创建文件名
        filename = f"demo_{age}_{gender}_{race}_{timestamp}.jpg"
        filepath = data_path / filename
        
        # 创建人工面部图像
        image = create_synthetic_face(age, gender, race)
        image.save(filepath, 'JPEG', quality=95)
        
        created_files.append(filename)
        
        if (i + 1) % 50 == 0:
            print(f"   已创建 {i + 1}/{num_samples} 个演示图像")
    
    print(f"✅ 演示图像创建完成")
    print(f"   创建数量: {len(created_files)}")
    print(f"   保存位置: {data_path.absolute()}")
    print(f"   文件格式: demo_[age]_[gender]_[race]_[timestamp].jpg")
    
    # 显示几个示例
    print(f"\n📋 示例文件:")
    for i, filename in enumerate(created_files[:5]):
        print(f"   {filename}")
    
    print(f"\n⚠️  重要提醒:")
    print(f"   - 这些是人工生成的演示图像，不是真实UTKFace数据")
    print(f"   - 仅用于演示脚本功能和CSV格式")
    print(f"   - 真实研究请使用官方UTKFace数据集")
    print(f"   - 下载地址: https://www.kaggle.com/datasets/jangedoo/utkface-new")
    
    return created_files

def create_synthetic_face(age: int, gender: int, race: int) -> Image.Image:
    """创建人工合成的面部图像"""
    
    # 创建128x128的RGB图像
    image = Image.new('RGB', (128, 128), color=(220, 200, 180))
    draw = ImageDraw.Draw(image)
    
    # 基于年龄、性别、种族调整颜色
    base_colors = {
        0: (255, 220, 177),  # 白人
        1: (139, 69, 19),    # 黑人
        2: (255, 228, 181),  # 亚洲人
        3: (160, 82, 45),    # 印度人
        4: (205, 133, 63),   # 其他
    }
    
    skin_color = base_colors.get(race, (200, 180, 160))
    
    # 年龄相关调整
    age_factor = age / 80.0
    skin_color = tuple(int(c * (1 - age_factor * 0.3)) for c in skin_color)
    
    # 绘制脸部轮廓
    face_bbox = [20, 15, 108, 113]
    draw.ellipse(face_bbox, fill=skin_color, outline=(100, 80, 60))
    
    # 眼睛
    eye_y = 45 + int(age_factor * 5)
    draw.ellipse([35, eye_y, 45, eye_y+8], fill=(255, 255, 255))
    draw.ellipse([83, eye_y, 93, eye_y+8], fill=(255, 255, 255))
    draw.ellipse([38, eye_y+2, 42, eye_y+6], fill=(50, 50, 50))
    draw.ellipse([86, eye_y+2, 90, eye_y+6], fill=(50, 50, 50))
    
    # 鼻子
    nose_y = 65 + int(age_factor * 3)
    draw.polygon([(64, nose_y), (60, nose_y+8), (68, nose_y+8)], fill=tuple(int(c*0.9) for c in skin_color))
    
    # 嘴巴
    mouth_y = 85 + int(age_factor * 5)
    mouth_width = 20 if gender == 0 else 18  # 女性嘴巴稍大
    draw.ellipse([64-mouth_width//2, mouth_y, 64+mouth_width//2, mouth_y+6], fill=(150, 50, 50))
    
    # 添加一些随机噪声使图像更真实
    pixels = np.array(image)
    noise = np.random.normal(0, 10, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 年龄相关的皱纹（简单线条）
    if age > 40:
        num_wrinkles = int((age - 40) / 10)
        for _ in range(num_wrinkles):
            x1 = random.randint(30, 98)
            y1 = random.randint(50, 90)
            x2 = x1 + random.randint(-10, 10)
            y2 = y1 + random.randint(-3, 3)
            draw.line([(x1, y1), (x2, y2)], fill=tuple(int(c*0.8) for c in skin_color), width=1)
    
    return Image.fromarray(pixels)

def main():
    """主函数"""
    print("🎭 UTKFace演示数据生成器")
    print("=" * 50)
    print("📝 用途：创建演示用的UTKFace格式图像")
    print("⚠️  注意：这不是真实的UTKFace数据集！")
    print("🔗 真实数据下载：https://www.kaggle.com/datasets/jangedoo/utkface-new")
    print("=" * 50)
    
    # 创建演示图像
    demo_files = create_demo_utkface_images("data", 150)
    
    print(f"\n🎉 演示数据创建完成！")
    print(f"📁 现在可以运行真实数据处理脚本测试功能")
    print(f"💡 运行命令: python src/manual_real_utkface_csv.py")
    
    return demo_files

if __name__ == "__main__":
    main() 