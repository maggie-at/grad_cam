import os  
import numpy as np  
from scipy.optimize import linear_sum_assignment  
  
def compute_iou(box1, box2):  
    # 计算IoU  
    inter_xmin = max(box1[0], box2[0])  
    inter_ymin = max(box1[1], box2[1])  
    inter_xmax = min(box1[2], box2[2])  
    inter_ymax = min(box1[3], box2[3])  
      
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)  
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  
    union_area = box1_area + box2_area - inter_area  
      
    return inter_area / union_area  
  
def read_annotation(file):  
    # 读取标注  
    with open(file, 'r') as f:  
        boxes = []  
        lines = f.readlines()  
        for line in lines:  
            box = list(map(float, line.strip().split(' ')[1:]))  
            boxes.append(box)  
        return boxes  
  
def get_miou(pred_dir, label_dir):  
    iou_dict = {}  
    for cls_dir in os.listdir(pred_dir):  
        iou_list = []  
        for pred_file in os.listdir(os.path.join(pred_dir, cls_dir)):  
            # 读取预测标注  
            pred_boxes = read_annotation(os.path.join(pred_dir, cls_dir, pred_file))  
              
            # 读取真实标注  
            label_boxes = read_annotation(os.path.join(label_dir, cls_dir, pred_file))  
              
            # 计算IoU矩阵  
            iou_matrix = np.zeros((len(pred_boxes), len(label_boxes)))  
            for i, pred_box in enumerate(pred_boxes):  
                for j, label_box in enumerate(label_boxes):  
                    iou_matrix[i, j] = compute_iou(pred_box, label_box)  
              
            # 使用匈牙利算法找到最大匹配  
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)  
              
            # 计算并保存匹配对的IoU  
            for i, j in zip(row_ind, col_ind):  
                iou_list.append(iou_matrix[i, j])  
  
        iou_dict[cls_dir] = np.mean(iou_list)  
  
    return iou_dict  
  
miou_dict = get_miou('predict', 'labels')  
for cls, miou in miou_dict.items():  
    print('mIoU for class {}: {}'.format(cls, miou))  
