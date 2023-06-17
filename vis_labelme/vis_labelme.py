import os
import json
from PIL import Image, ImageDraw
import numpy as np
import labelme

# 输入文件夹路径
DATA_DIR = '../data/eggs'
folder_path = os.path.join(DATA_DIR, 'images')

# 输出文件夹路径
output_folder = os.path.join(DATA_DIR, 'out_labelme')
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的标注文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        # 加载 Labelme 格式的标注文件
        annotation_file = os.path.join(folder_path, filename)
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # 获取图像路径和标注信息
        image_path = os.path.join(folder_path, data['imagePath'])
        image = Image.open(image_path)
        shapes = data['shapes']

        # 创建空白画布
        image_pil = image.copy()
        draw = ImageDraw.Draw(image_pil)

        # 绘制标注区域
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            points = np.array(points).reshape((-1, 2)).flatten().tolist()
            print(points,flush=True)
            draw.polygon(points, outline='red')
            # draw.text(points[0], label, fill='red')

        # 保存可视化后的图片
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
        image_pil.save(output_path)