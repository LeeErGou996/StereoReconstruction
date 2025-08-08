# -*- coding: utf-8 -*-
"""
Create 1x3 Comparison Image - Jupyter Notebook Version
将rectified、without、gt_disparity_color三张图绘制为1*3的联合图

使用方法：
1. 在Jupyter notebook中运行第一个单元格导入库
2. 运行第二个单元格定义函数
3. 运行第三个单元格执行主程序
"""

# ============================================================================
# 单元格 1: 导入必要的库
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 库导入完成")

# ============================================================================
# 单元格 2: 定义函数
# ============================================================================

def load_and_resize_image(image_path, target_size=None):
    """
    加载并调整图像尺寸
    
    Parameters:
    -----------
    image_path : str
        图像文件路径
    target_size : tuple, optional
        目标尺寸 (width, height)
    
    Returns:
    --------
    numpy.ndarray : 加载的图像数组
    """
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    
    # 加载图像
    img = mpimg.imread(image_path)
    print(f"✅ 加载图像: {image_path}")
    print(f"   原始尺寸: {img.shape}")
    
    # 如果指定了目标尺寸，调整图像大小
    if target_size is not None:
        from PIL import Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        img = np.array(pil_img) / 255.0
        print(f"   调整后尺寸: {img.shape}")
    
    return img

def create_1x3_comparison(image_paths, titles, output_path, figsize=(18, 6), dpi=300):
    """
    创建1*3的联合图
    
    Parameters:
    -----------
    image_paths : list
        三个图像文件的路径列表
    titles : list
        三个图像的标题列表
    output_path : str
        输出文件路径
    figsize : tuple
        图像尺寸
    dpi : int
        图像分辨率
    """
    
    # 检查输入参数
    if len(image_paths) != 3 or len(titles) != 3:
        print("❌ 错误: 需要3个图像路径和3个标题")
        return False
    
    # 检查所有图像文件是否存在
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"❌ 图像文件不存在: {path}")
            return False
    
    print("=== 开始创建1*3联合图 ===")
    print(f"图像文件:")
    for i, path in enumerate(image_paths):
        print(f"  {i+1}. {path}")
    print(f"标题:")
    for i, title in enumerate(titles):
        print(f"  {i+1}. {title}")
    
    # 加载所有图像
    images = []
    for path in image_paths:
        img = load_and_resize_image(path)
        if img is None:
            return False
        images.append(img)
    
    # 获取所有图像的尺寸信息
    shapes = [img.shape for img in images]
    print(f"\n图像尺寸信息:")
    for i, shape in enumerate(shapes):
        print(f"  {i+1}. {shape}")
    
    # 创建1*3的子图
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('图像对比分析', fontsize=16, fontweight='bold', y=0.95)
    
    # 绘制每个图像
    for i, (img, title, ax) in enumerate(zip(images, titles, axes)):
        # 显示图像
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA图像
            # 转换为RGB
            rgb_img = img[:, :, :3]
            ax.imshow(rgb_img)
        else:
            ax.imshow(img)
        
        # 设置标题
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        
        # 移除坐标轴
        ax.axis('off')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        
        print(f"✅ 绘制图像 {i+1}: {title}")
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ 联合图已保存到: {output_path}")
    
    # 显示图像
    plt.show()
    
    return True

print("✅ 函数定义完成")

# ============================================================================
# 单元格 3: 主程序执行
# ============================================================================

# 显示当前工作目录
print("=== 1*3图像联合图生成器 ===")
print(f"当前工作目录: {os.getcwd()}")

# 检查当前目录下的文件
print("\n当前目录下的PNG文件:")
png_files = [f for f in os.listdir('.') if f.endswith('.png')]
for i, file in enumerate(sorted(png_files), 1):
    print(f"  {i}. {file}")

# 定义图像文件路径
image_paths = [
    "rectified.png",           # 校正后的图像
    "without.png",             # 未校正的图像
    "gt_disparity_color.png"   # 真实视差图
]

# 定义图像标题
titles = [
    "Rectified Image\n(校正后图像)",
    "Original Image\n(原始图像)", 
    "Ground Truth Disparity\n(真实视差图)"
]

# 输出文件路径
output_path = "1x3_comparison.png"

# 创建联合图
success = create_1x3_comparison(
    image_paths=image_paths,
    titles=titles,
    output_path=output_path,
    figsize=(18, 6),
    dpi=300
)

if success:
    print("\n=== 处理完成 ===")
    print(f"生成的联合图: {output_path}")
    print("图像包含:")
    for i, title in enumerate(titles):
        print(f"  {i+1}. {title}")
else:
    print("\n❌ 处理失败")

# ============================================================================
# 单元格 4: 可选 - 自定义参数
# ============================================================================

# 如果您想要自定义参数，可以修改以下变量并重新运行

# 自定义图像路径（如果需要）
# custom_image_paths = [
#     "your_image1.png",
#     "your_image2.png", 
#     "your_image3.png"
# ]

# 自定义标题（如果需要）
# custom_titles = [
#     "Custom Title 1",
#     "Custom Title 2",
#     "Custom Title 3"
# ]

# 自定义输出文件名（如果需要）
# custom_output_path = "custom_comparison.png"

# 自定义图像尺寸（如果需要）
# custom_figsize = (20, 8)  # 更宽的图像
# custom_dpi = 300          # 更高的分辨率

print("✅ 程序执行完成") 