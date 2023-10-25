import os
import numpy as np
from tqdm import tqdm

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import models, transforms

from utils.layer_utils import GradCAM, show_cam_on_image

cls_dict={'class_name_0': 0, 
          'class_name_10': 1, 
          'class_name_2': 2, 
          'class_name_3': 3, 
          'class_name_7': 4, 
          'class_name_8': 5}

def cnn_cam(data_path, res_path, num_classes):
    model = models.resnet34()
    in_channel = model.fc.in_features
    model.fc = torch.nn.Linear(in_channel, num_classes)
    model.load_state_dict(torch.load('./train_model/resnet/resNet34-hegang.pth', map_location='cpu'))
    
    target_layers = [model.layer2, model.layer3, model.layer4]

    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    for cls_folder in os.listdir(data_path):  
        cls_folder_path = os.path.join(data_path, cls_folder)  
        # 在结果目录中创建一个与类别相同的文件夹  
        res_folder_path = os.path.join(res_path, cls_folder)
        os.makedirs(res_folder_path, exist_ok=True)
        os.makedirs(os.path.join(res_folder_path, "gray/"), exist_ok=True)
        os.makedirs(os.path.join(res_folder_path, "bbox/"), exist_ok=True)
        os.makedirs(os.path.join(res_folder_path, "label/"), exist_ok=True)

        target_category = cls_dict[cls_folder]

        for img_file in tqdm(os.listdir(cls_folder_path), desc="Processing class "+cls_folder):
            img_path = os.path.join(cls_folder_path, img_file)
            # load image
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)

            # [C, H, W]
            img_tensor = data_transform(img)
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                            grayscale_cam,
                                            use_rgb=True)
            plt.imsave(os.path.join(res_folder_path, "gray/", img_file), visualization)
            
            grayscale_cam_8bit = (grayscale_cam * 255).astype('uint8')
            
            # 设置每个类别的分割阈值
            low = 80
            mid = 127
            high = 150
            thresh = [high, mid, mid, mid, high, mid]
            _, thresh = cv2.threshold(grayscale_cam_8bit, thresh[cls_dict[cls_folder]], 255, cv2.THRESH_BINARY)
            
            # 画出预测框
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                # 忽略面积小于64的预测框
                if area >= 64:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(res_folder_path, "bbox/", img_file), img)

            # save bbox coordinates to txt file
            with open(os.path.join(res_folder_path, "label/", img_file.split('.')[0] + '.txt'), 'w') as f:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    f.write(f"{x}, {y}, {x+w}, {y+h}\n")


if __name__ == '__main__':
    cnn_cam(data_path='/home/huyu/codes/exp1/HEGANG/val/', 
            res_path='/home/huyu/codes/exp1/HEGANG/Multi_CLS/predict/',
            num_classes=6)
