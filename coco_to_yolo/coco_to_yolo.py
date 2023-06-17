import json

def convert_coco_to_yolo(annotation_file, output_dir):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    categories = {category['id']: category['name'] for category in data['categories']}
    
    for image in data['images']:
        image_id = image['id']
        image_width = image['width']
        image_height = image['height']
        file_name = image['file_name']
        
        yolo_labels = []
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                category_name = categories[category_id]
                bbox = annotation['bbox']
                
                x = (bbox[0] + bbox[2] / 2) / image_width
                y = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
                
                yolo_label = f"{category_id} {x} {y} {width} {height}"
                yolo_labels.append(yolo_label)
        
        # Save YOLO labels to a file
        output_file = f"{output_dir}/{file_name[:-4]}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_labels))

# Example usage
import os
if __name__ == '__main__':
    DATA_DIR = '../data/eggs'
    annotation_file = os.path.join(DATA_DIR, 'Annotations/coco_info.json')
    output_dir = os.path.join(DATA_DIR, 'yolo')
    os.makedirs(output_dir, exist_ok=True)
    convert_coco_to_yolo(annotation_file, output_dir)