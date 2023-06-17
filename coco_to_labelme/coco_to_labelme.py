import json
import os

label_names=['desiccative', 'impurity']
DATA_DIR = '../data/bubbles'

with open(os.path.join(DATA_DIR, 'Annotations/coco_info.json'), 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

for image_data in coco_data['images']:
    image_id = image_data['id']
    file_name = image_data['file_name']
    image_annotations = []

    # 查找当前图像的所有标注
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            image_annotations.append(annotation)

    # 构建对应的Labelme数据
    labelme_data = {
        'version': '4.5.6',
        'flags': {},
        'shapes': [],
        'imagePath': file_name,
        'imageData': None
    }

    # 构建每个标注的形状
    for annotation in image_annotations:
        shape = {
            'label': label_names[annotation['category_id']],
            'points': [],  # 填充顶点坐标
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
        }

        # 将COCO格式的边界框坐标转换为多边形坐标
        x, y, width, height = annotation['bbox']
        x_min, y_min = x, y
        x_max, y_max = x + width, y + height
        shape['points'] = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]

        labelme_data['shapes'].append(shape)

    # 保存为单独的Labelme格式的JSON文件
    labelme_filename = file_name.replace('.png', '.json')  # 可根据需要调整文件名
    with open(os.path.join(DATA_DIR, 'Images', labelme_filename), 'w') as f:
        json.dump(labelme_data, f)
    # break