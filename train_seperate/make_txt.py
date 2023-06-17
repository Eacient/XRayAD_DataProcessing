import os
import json

def is_anomaly(json_data):
    if len (json_data['shapes']) == 0:
        return False
    else:
        return True


DATA_DIR = '../data/eggs'
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

            if (is_anomaly(json_data)):
                anomaly_data_list.append(json_data['imagePath'])
            else:
                normal_data_list.append(json_data['imagePath'])
            all_data_list.append(json_data['imagePath'])


# 保存 all.json
all_filename = "all.txt"
all_filepath = os.path.join(DATA_DIR, all_filename)
with open(all_filepath, "w") as f:
    for j in all_data_list:
        f.write(j+'\n')

# 保存 normal.json
normal_filename = "normal.txt"
normal_filepath = os.path.join(DATA_DIR, normal_filename)
with open(normal_filepath, "w") as f:
    for j in normal_data_list:
        f.write(j+'\n')

# 保存 anomaly.json
anomaly_filename = 'anomaly.txt'
anomaly_filepath = os.path.join(DATA_DIR, anomaly_filename)
with open(anomaly_filepath, 'w') as f:
    for j in anomaly_data_list:
        f.write(j+'\n')

# 保存 train.txt 和 val.txt
train_filename = 'train.txt'
val_filename = 'val.txt'
train_filepath = os.path.join(DATA_DIR, train_filename)
val_filepath = os.path.join(DATA_DIR, val_filename)
train_ratio = 0.9
with open(train_filepath, 'w') as f:
    for i in range(int(len(normal_data_list) * 0.9)):
        f.write(normal_data_list[i]+'\n')
with open(val_filepath, 'w') as f:
    for i in range(int(len(normal_data_list) * 0.9), len(normal_data_list)):
        f.write(normal_data_list[i] + '\n')
    for j in anomaly_data_list:
        f.write(j+'\n')
