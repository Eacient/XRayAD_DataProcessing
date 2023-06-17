import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure, morphology
import json
import os

def convert_to_labelme(gt_box_list, pred_box_list, image_name, json_dir, height, width):
    data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_name+'.png',
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    # 添加事实框
    for gt_box in gt_box_list:
        x1, y1, x2, y2 = gt_box
        shape = {
            "label": "gt_box",
            "points": [[y1*width, x1*height], [y2*width, x2*height]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        data["shapes"].append(shape)

    # 添加预测框
    for pred_box in pred_box_list:
        x1, y1, x2, y2 = pred_box
        shape = {
            "label": "pred_box",
            "points": [[y1*width, x1*height], [y2*width, x2*height]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        data["shapes"].append(shape)

    # 将数据保存为JSON文件
    json_data = json.dumps(data, indent=4)
    output_file = f"{image_name}.json"
    with open(os.path.join(json_dir,output_file), "w") as f:
        f.write(json_data)

    # print(f"转换完成，已保存为{output_file}")


def convert_score_to_bbox(score_map, threshold, min_size=None):
    # 将得分映射转换为二值掩码
    binary_mask = (score_map >= threshold).astype(int)

    # 删掉小于min_size(像素)的目标
    if min_size:
        mask_without_small = morphology.remove_small_objects(binary_mask, min_size=min_size, connectivity=2)
        binary_mask = mask_without_small
    
    # 使用形态学操作将二值掩码转换为目标预测框
    labeled_image, num_objects = measure.label(binary_mask, connectivity=2, return_num=True)
    regions = measure.regionprops(labeled_image)
    bounding_boxes = [list(region.bbox) for region in regions]

    # 坐标归一化到0-1范围内，与yolo标注对应
    image_height, image_width = score_map.shape
    for box in bounding_boxes:
        box[0] = box[0] / image_height
        box[1] = box[1] / image_width
        box[2] = box[2] / image_height
        box[3] = box[3] / image_width
    # print(bounding_boxes)
    return bounding_boxes

def calculate_iou(bbox1, bbox2):
    # 计算两个边界框的交并比（IoU）
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

# count tp, fp, fn
def calculate_metrics(y_true, prediction_boxes, iou_thresh=0.00001):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched = set()
    
    # print(y_true, prediction_boxes)
    for true_box in y_true:
        iou_values = [calculate_iou(true_box, pred_box) for pred_box in prediction_boxes]
        # print(iou_values)
        max_iou = max(iou_values) if iou_values else 0
        
        if max_iou >= iou_thresh:
            index = iou_values.index(max_iou)
            if index not in matched:
                true_positives += 1
                matched.add(index)
            else:
                # false_positives += 1
                true_positives += 1
        else:
            false_negatives += 1
    
    false_positives += len(prediction_boxes) - len(matched)
    
    return true_positives, false_positives, false_negatives

def calculate_precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0
    return true_positives / (true_positives + false_positives)

def calculate_recall(true_positives, false_negatives):
    if true_positives + false_negatives == 0:
        return 0
    return true_positives / (true_positives + false_negatives)

# calculate precision, recall at a specific confidence score
def calculate_precision_recall(y_true_list, y_score_list, threshold, filename_list=None, height_list=None, width_list=None):
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0

    if filename_list is not None:
        for y_true, y_score, filename, height, width in zip(y_true_list, y_score_list, filename_list, height_list, width_list):
            # 将得分映射转换为目标检测框
            prediction_boxes = convert_score_to_bbox(y_score, threshold)

            convert_to_labelme(y_true, prediction_boxes, filename, '/root/autodl-tmp/labelme_anno', height, width)
            
            true_positives, false_positives, false_negatives = calculate_metrics(y_true, prediction_boxes)
            
            all_true_positives += true_positives
            all_false_positives += false_positives
            all_false_negatives += false_negatives
    else:
        for y_true, y_score in zip(y_true_list, y_score_list):
            # 将得分映射转换为目标检测框
            prediction_boxes = convert_score_to_bbox(y_score, threshold)
            
            true_positives, false_positives, false_negatives = calculate_metrics(y_true, prediction_boxes)
            
            all_true_positives += true_positives
            all_false_positives += false_positives
            all_false_negatives += false_negatives
    
    precision = calculate_precision(all_true_positives, all_false_positives)
    recall = calculate_recall(all_true_positives, all_false_negatives)
    
    return precision, recall

def calculate_auc(precisions, recalls):
    return auc(recalls, precisions)

# calculate precision,recall at different confidence score and area-under-curve
def calculate_average_precision(y_true_list, y_score_list, cls_name):
    thresholds = np.linspace(0, 1, num=101)  # 阈值范围为0到1，共101个阈值
    precisions = []
    recalls = []
    
    # go through different confidence threshold
    for threshold in thresholds:
        precision, recall = calculate_precision_recall(y_true_list, y_score_list, threshold)
        print(threshold, precision, recall)
        precisions.append(precision)
        recalls.append(recall)

    # calculate area under curve
    ap = calculate_auc(precisions, recalls)
    plot_precision_recall_curve(precisions, recalls, cls_name)
    plot_precision_thresh_curve(precisions, thresholds, cls_name)
    plot_recall_thresh_curve(recalls, thresholds, cls_name)
    return ap

# 用于可视化
def plot_precision_recall_curve(precisions, recalls, cls_name):
    plt.figure()
    plt.plot(recalls, precisions, '-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
    plt.savefig(f'{cls_name}_pr.png')
def plot_precision_thresh_curve(precisions, thresholds, cls_name):
    plt.figure()
    plt.plot(thresholds, precisions, '-o')
    plt.xlabel('thresholds')
    plt.ylabel('Precision')
    plt.title('Precision-Threshold Curve')
    plt.grid(True)
    plt.show()
    plt.savefig(f'{cls_name}_p.png')
def plot_recall_thresh_curve(recalls, thresholds, cls_name):
    plt.figure()
    plt.plot(thresholds, recalls, '-o')
    plt.xlabel('thresholds')
    plt.ylabel('recall')
    plt.title('Recall-Threshold Curve')
    plt.grid(True)
    plt.show()
    plt.savefig(f'{cls_name}_r.png')

def parse_yolo_annotation(anno_file_path, image_width, image_height):
    with open(anno_file_path, 'r') as file:
        lines = file.readlines()

    y_true = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        
        # 将yolo坐标转化为绝对坐标
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height

        y_true.append([y_min, x_min, y_max, x_max])

        # # Append the bounding box information to the y_true list
        # y_true.append({'class_id': int(class_id), 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max})
    
    return y_true

# require the dataset return image_names
def load_yolo_annotations(yolo_dir, image_names, image_width, image_height):
    y_true_list = []
    for image_name in image_names:
        annotation_file = os.path.join(yolo_dir, image_name + '.txt')
        y_true = parse_yolo_annotation(annotation_file, image_width, image_height)
        y_true_list.append(y_true)
    return y_true_list

