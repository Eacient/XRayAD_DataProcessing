import os
import json

def convert_annotation(json_data, source_folder, annotation_folder, name='can'):
    # filename = f"{name}/{source_folder}/{'.'.join(json_data['imagePath'].split('.')[:-1])}.jpg"
    filename = f"{'.'.join(json_data['imagePath'].split('.')[:-1])}.jpg"
    clsname = name
    
    if len(json_data["shapes"]) == 0:
        # 无异常样本
        label = 0
        label_name = "good"
        return {
            "filename": filename,
            "label": label,
            "label_name": label_name,
            "clsname": clsname
        }
    else:
        # 异常样本
        label = 1
        label_name = "defective"
        maskname = f"{name}/{annotation_folder}/{json_data['imagePath']}"
        return {
            "filename": filename,
            "label": label,
            "label_name": label_name,
            "maskname": maskname,
            "clsname": clsname
        }



DATA_DIR = '../data/cans'
# 定义源文件夹路径和标注文件夹路径
source_folder = os.path.join(DATA_DIR, "voc/JPEGImages")
annotation_folder = os.path.join(DATA_DIR, "voc/SegmentationClassPNG")
# 定义存放labelme JSON文件的文件夹路径
json_folder = os.path.join(DATA_DIR, "Images")
# 存放转换后的标注数据的列表
all_data_list = []
normal_data_list = []
anomaly_data_list = []
# 遍历文件夹中的JSON文件
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        # 读取JSON文件
        with open(os.path.join(json_folder, filename)) as f:
            json_data = json.load(f)
        
    # 转换标注数据
        converted_data = convert_annotation(json_data, source_folder, annotation_folder)
        
        if converted_data["label"] == 0:
            # 无异常样本，添加到 train_data_list
            normal_data_list.append(converted_data)
        else:
            anomaly_data_list.append(converted_data)
        
        # 添加到 test_data_list
        all_data_list.append(converted_data)
# 将转换后的标注数据保存为 all.json
all_filename = "all.json"
all_filepath = os.path.join(DATA_DIR, all_filename)
with open(all_filepath, "w") as f:
    for j in all_data_list:
        f.write(json.dumps(j)+'\n')
# 将转换后的标注数据保存为 normal.json
train_filename = "normal.json"
train_filepath = os.path.join(DATA_DIR, train_filename)
with open(train_filepath, "w") as f:
    for j in normal_data_list:
        f.write(json.dumps(j)+'\n')
# 将转换后的标注数据保存为 anomaly.json
anomaly_filename = 'anomaly.json'
anomaly_filepath = os.path.join(DATA_DIR, anomaly_filename)
with open(anomaly_filepath, 'w') as f:
    for j in anomaly_data_list:
        f.write(json.dumps(j)+'\n')