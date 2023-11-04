import os  
import numpy as np  
from tqdm import tqdm
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torchvision.transforms as transforms  
from torchvision.models import resnet34  
import cv2
from PIL import Image  
import matplotlib.pyplot as plt  


'''
小论文 - CAM代码
'''
class CamModel(nn.Module):  
    def __init__(self, model):  
        super(CamModel, self).__init__()  
        self.features = nn.Sequential(*list(model.children())[:-2])  # 提取除全连接层之外的所有层  
        self.fc = model.fc  # 全连接层  

    def forward(self, x):  
        x = self.features(x)  
        self.featuremap = x.detach()  
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)  # 全局平均池化  
        x = self.fc(x)  
        return x  
  
    def get_cam(self, weight):  
        cam = F.conv2d(self.featuremap, weight)  
        cam = F.relu(cam)  
        # print(cam)
        # cam = cam.view(cam.size(0), -1)  
        # cam -= cam.min(dim=1, keepdim=True)[0]  
        # cam /= cam.max(dim=1, keepdim=True)[0]  
        return cam  


# 定义预处理  
data_transform = transforms.Compose([transforms.ToTensor(),  
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  
  
# 加载模型  
model = resnet34()  
in_channel = model.fc.in_features  
num_classes = 6  
model.fc = nn.Linear(in_channel, num_classes)  
model.load_state_dict(torch.load('./train_model/resnet/resNet34-neu.pth', map_location='cpu'))  
  
# 创建CAM模型  
cam_model = CamModel(model)  
  
# 遍历"val/"文件夹下的所有图片  
data_path = '/home/ubuntu/workspace/hy/dataset/NEU-DET/val/'  # 图片文件夹路径  

res_path = "/home/ubuntu/workspace/hy/dataset/NEU-DET/predict_cam/"


for cls_name in os.listdir(data_path):  
    cls_folder_path = os.path.join(data_path, cls_name)  
    # 在结果目录中创建一个与类别相同的文件夹  
    res_folder_path = os.path.join(res_path, cls_name)
    os.makedirs(res_folder_path, exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "gray/"), exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "bbox/"), exist_ok=True)
    os.makedirs(os.path.join(res_folder_path, "label/"), exist_ok=True)

    for img_file in tqdm(os.listdir(cls_folder_path), desc="Processing class "+cls_name):  
        image_path = os.path.join(cls_folder_path, img_file)    
        
        img = Image.open(image_path)    
        img = data_transform(img).unsqueeze(0)    
            
        # 计算输出和CAM    
        output = cam_model(img)    
        _, predicted = torch.max(output, 1)    
        weight = cam_model.fc.weight[predicted, :].unsqueeze(-1).unsqueeze(-1)    
        cam = cam_model.get_cam(weight)    
        
        # 将矩阵放大到原图大小    
        matrix=cam.squeeze()  
        matrix_resized = F.interpolate(matrix.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)    
        matrix_resized = matrix_resized.squeeze(0).squeeze(0).detach().numpy()    
        
        # 定义形态学结构元素，例如一个 5x5 的方形  
        kernel = np.ones((13,13),np.uint8) 

        # 将 matrix_resized 转换为 0-255 的 8 位整数  
        matrix_resized = (matrix_resized * 255).astype(np.uint8)  
        
        # 在 matrix_resized 上执行闭合操作  
        matrix_resized = cv2.morphologyEx(matrix_resized, cv2.MORPH_CLOSE, kernel)  

        # 在 matrix_resized 上执行阈值分割  
        ret, thresh = cv2.threshold(matrix_resized, 127, 255, 0)  
        
        # 寻找阈值图像中的轮廓  
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
        
        # 对于每个轮廓，获取其边界框并将其写入到 txt 文件中  
        with open(os.path.join(res_folder_path, "label/", img_file.split('.')[0]+".txt"), 'w') as f:  
            for contour in contours:  
                x, y, w, h = cv2.boundingRect(contour)  
                # 格式化为 "x_min y_min x_max y_max"  
                bbox = f"{x} {y} {x+w} {y+h}\n"  
                f.write(bbox)  

        # 生成热力图    
        plt.imshow(matrix_resized)
        plt.axis('off')
        plt.savefig(os.path.join(res_folder_path, "gray/", img_file), bbox_inches='tight', pad_inches = 0)    
        