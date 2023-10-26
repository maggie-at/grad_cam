import numpy as np
import os

def calculate_iou(boxA, boxB):
    interArea = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0])) * max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_metrics(pred_dir, label_dir, iou_threshold=0.4):
    TP = 0
    FP = 0
    FN = 0
    ious = []
    for filename in os.listdir(pred_dir):
        pred_file = os.path.join(pred_dir, filename)
        label_file = os.path.join(label_dir, filename)
        if os.path.isfile(pred_file) and os.path.isfile(label_file):
            with open(pred_file, 'r') as f:
                pred_boxes = [list(map(int, line.strip().split(' '))) for line in f]
            with open(label_file, 'r') as f:
                label_box = list(map(int, f.readline().strip().split(' ')))
            if pred_boxes:
                ious_individual = [calculate_iou(pred_box, label_box) for pred_box in pred_boxes]
                ious.append(max(ious_individual))
                TP += sum(iou >= iou_threshold for iou in ious_individual)
                FP += sum(iou < iou_threshold for iou in ious_individual)
            else:
                FN += 1
    miou = np.mean(ious)
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    return miou, recall, precision


pred_dir = '/home/ubuntu/workspace/hy/dataset/DAGM/result/predict/label/'
label_dir = '/home/ubuntu/workspace/hy/dataset/DAGM/label/val'
miou, recall, precision = calculate_metrics(pred_dir, label_dir)  
print('Mean IoU:', miou)  
print('Recall:', recall)  
print('Precision:', precision)  
