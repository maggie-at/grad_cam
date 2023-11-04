import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(box1, box2):
    print(box1, box2)
    # 计算 IoU
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def read_truth_annotation(file):
    # 读取标注
    with open(file, 'r') as f:
        boxes = []
        lines = f.readlines()
        for line in lines:
            box = list(map(float, line.strip().split(' ')[1:]))
            
            swapped_list = box[:]
            swapped_list[1] = box[3]
            swapped_list[3] = box[1]
            boxes.append(swapped_list)
        return boxes


def read_pred_annotation(file):
    # 读取标注
    with open(file, 'r') as f:
        boxes = []
        lines = f.readlines()
        for line in lines:
            box = list(map(float, line.strip().split(' ')[0:]))
            boxes.append(box)
        return boxes

def calculate_mIoU_and_Gt_Known_Loc_Acc(pred_dir, label_dir, iou_threshold=0.5):
    iou_list = []
    correct_loc_count = 0
    total_samples = 0
    
    for cls_dir in os.listdir(pred_dir):
        for pred_file in os.listdir(os.path.join(pred_dir, cls_dir, "label/")):
            # 读取预测标注
            pred_boxes = read_pred_annotation(os.path.join(pred_dir, cls_dir, "label/", pred_file))

            # 读取真实标注
            label_boxes = read_truth_annotation(os.path.join(label_dir, pred_file))

            # 计算 IoU 矩阵
            iou_matrix = np.zeros((len(pred_boxes), len(label_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, label_box in enumerate(label_boxes):
                    iou_matrix[i, j] = compute_iou(pred_box, label_box)

            # 使用匈牙利算法找到最大匹配
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            # 计算并保存匹配对的 IoU
            for i, j in zip(row_ind, col_ind):
                iou = iou_matrix[i, j]
                iou_list.append(iou)
                if iou >= iou_threshold:
                    correct_loc_count += 1
            total_samples += len(pred_boxes)
    
    # 计算 mIoU
    mIoU = np.mean(iou_list)
    
    # 计算 Gt-Known Loc. Acc
    gt_known_loc_acc = correct_loc_count / total_samples

    return mIoU, gt_known_loc_acc


pred_directory = "/home/ubuntu/workspace/hy/dataset/NEU-DET/predict_grad_ei/"
label_directory = "/home/ubuntu/workspace/hy/dataset/NEU-DET/label/"
iou_threshold = 0.5
mIoU, gt_known_loc_acc = calculate_mIoU_and_Gt_Known_Loc_Acc(pred_directory, label_directory, iou_threshold)
print(f"mIoU (IoU >= {iou_threshold}): {mIoU}")
print(f"Gt-Known Loc. Acc (IoU >= {iou_threshold}): {gt_known_loc_acc}")
