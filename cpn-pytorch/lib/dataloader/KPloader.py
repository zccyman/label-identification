import cv2
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from utils.create_csv import create_csv

# 定义平移translate函数
def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
 
    # 返回转换后的图像
    return shifted

def rotate_point(x0, y0, center_x, center_y, angle, scale = 1.0):
    cos = math.cos(angle * (np.pi / 180.0))
    sin = math.sin(angle * (np.pi / 180.0))
    delta_x = x0 - center_x
    delta_y = y0 - center_y
    x = delta_x * cos + delta_y * sin + center_x
    y = -delta_x * sin + delta_y * cos + center_y
    x = math.floor(x)
    y = math.floor(y)
    
    return x, y

def rotate_image_keep_size(img, center_x, center_y, angle, scale = 1.0):
    height, width = img.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated_mat = cv2.warpAffine(img, rotation_mat, (width, height))

    return rotated_mat

def data_augmentation(image, keypoints, operation = 0):
    kyp = copy.deepcopy(keypoints)
    keypoint_num = len(kyp)
    #print("keypoint_num: ", keypoint_num)
    
    global image_augment
    
    if 0 == operation: #h
        map_talbel = list(range(20, 0, -1)) + list(range(80, 60, -1)) + \
        list(range(60, 40, -1)) + list(range(40, 20, -1))
        #print(map_talbel)
        
        #print("keypoint_num:", len(kyp))
        for k in range(0, keypoint_num):
            keypoints[k, 0] = image.shape[1] - kyp[map_talbel[k] - 1, 0]
            keypoints[k, 1] = kyp[map_talbel[k] - 1, 1] 
                                
        image_augment = cv2.flip(image, 1)
        
        #zoom
        #scale = 1.0 + 0.5 * (random.random() - 0.5)
        #for k in range(0, keypoint_num):
        #    keypoints[k, 0] = int(scale * kyp[k, 0])
        #    keypoints[k, 1] = int(scale * kyp[k, 1])
                        
        #image_augment = cv2.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
        
    elif 1 == operation:#v
        map_talbel = list(range(60, 40, -1)) + list(range(40, 20, -1)) + \
        list(range(20, 0, -1)) + list(range(80, 60, -1))
        
        for k in range(0, keypoint_num):
            keypoints[k, 0] = kyp[map_talbel[k] - 1, 0]
            keypoints[k, 1] = image.shape[0] - kyp[map_talbel[k] - 1, 1]
            
        image_augment = cv2.flip(image, 0)  
        
        #zoom
        #scale = 1.0 + 0.5 * (random.random() - 0.5)
        #for k in range(0, keypoint_num):
        #    keypoints[k, 0] = int(scale * kyp[k, 0])
        #    keypoints[k, 1] = int(scale * kyp[k, 1])
                        
        #image_augment = cv2.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
        
    elif 2 == operation:#hv
        map_talbel = list(range(41, 61, 1)) + list(range(61, 81, 1)) + \
        list(range(1, 21, 1)) + list(range(21, 41, 1))
        
        for k in range(0, keypoint_num):
            keypoints[k, 0] = image.shape[1] - kyp[map_talbel[k] - 1, 0]
            keypoints[k, 1] = image.shape[0] - kyp[map_talbel[k] - 1, 1]
            
        image_augment = cv2.flip(image, -1)
        
        #zoom
        #scale = 1.0 + 0.5 * (random.random() - 0.5)
        #for k in range(0, keypoint_num):
        #    keypoints[k, 0] = int(scale * kyp[k, 0])
        #    keypoints[k, 1] = int(scale * kyp[k, 1])
                        
        #image_augment = cv2.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
        
    elif 3 == operation:#rotation
        height, width = image.shape[:2]
        center_x, center_y = (width / 2, height / 2)
        
        angle = random.randint(-1, 1)
        
        image_augment = rotate_image_keep_size(image, center_x, center_y, angle)
        
        for k in range(0, keypoint_num):
            keypoints[k, 0], keypoints[k, 1] = rotate_point(kyp[k, 0], kyp[k, 1], \
            center_x, center_y, angle)
            
    elif 4 == operation:#translate
        shift_x = random.randint(-25, 25)
        shift_y = random.randint(-25, 25)
        
        image_augment = translate(image, shift_x, shift_y)
        
        for k in range(0, keypoint_num):
            keypoints[k, 0] = kyp[k, 0] + shift_x
            keypoints[k, 1] = kyp[k, 1] + shift_y
                                
    return image_augment, keypoints

class KPloader(data.Dataset):
    def __init__(self, cfg):
        self.vis = cfg.vis
        self.is_train = cfg.is_train
        self.num_class = cfg.num_class
        
        self.featuremap_scale = cfg.featuremap_scale
        self.out_res = cfg.output_shape
        self.data_shape = cfg.data_shape
        
        self.pixel_means = cfg.pixel_means
        
        self.img_path = cfg.img_path
        self.csv_file_path = cfg.csv_file_path
        
        #gk
        self.gk15 = cfg.gk15
        self.gk11 = cfg.gk11
        self.gk9 = cfg.gk9
        self.gk7 = cfg.gk7      
        
        #create csv
        create_csv(self.img_path, self.csv_file_path, self.num_class, self.is_train)
        
        self.csv_file = pd.read_csv(self.csv_file_path)
        #self.csv_file = self.csv_file.rename(columns=lambda x: x.replace('[','').replace(']','').replace('"',''))
        #print(self.csv_file)
        
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        image_name = self.csv_file["image_name"][index]
        #print(image_name)
        if self.is_train:
            gt_bbox = self.csv_file["gt_bbox"][index]
            keypoint = []
            for i in range(self.num_class):
                x = self.csv_file["keypoints"][index].split(" ")[i * 3]
                y = self.csv_file["keypoints"][index].split(" ")[i * 3 + 1]
                valid = self.csv_file["keypoints"][index].split(" ")[i * 3 + 2]
                keypoint.append(x)
                keypoint.append(y)
                keypoint.append(valid)
            
            keypoint = np.array(keypoint).reshape(self.num_class, 3).astype(np.float32)
            
        image = cv2.imread(os.path.join(self.img_path, image_name), 1)
        #print("image_w:", image.shape[1], "image_h:", image.shape[0])
        
        if self.is_train:
            image, keypoint = data_augmentation(image, keypoint, random.randint(0, 3)) #flip and rotation augment
            
            #keypoint scale
            keypoint[:, 0] //= float(image.shape[1] / self.data_shape[1])  #x
            keypoint[:, 1] //= float(image.shape[0] / self.data_shape[0])  #y

        if 2 == image.ndim:
            #print("GRAY")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif 3 == image.ndim:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(image, (self.data_shape[1], self.data_shape[0]))#w h
        #print(image_name, keypoint, keypoint.shape[0], keypoint.shape[1])
        
        img = im_to_torch(img)
        img = color_normalize(img, self.pixel_means)
        
        if self.vis and self.is_train:
            for i in range(self.num_class):
                cv2.circle(image, (keypoint[i][0], keypoint[i][1]), 2, (0, 0, 255), -1) #(x, y)
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
            #cv2.destroyAllWindows()

        if self.is_train:
            keypoint[:, :2] //= int(self.featuremap_scale) # output size is 1/4 input size
            pts = torch.Tensor(keypoint)
            #print(keypoint[:, 0], keypoint[:, 1], keypoint[:, 2])
            #print(pts[:, 0], pts[:, 1], pts[:, 2])
            
            target15 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target11 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target9 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target7 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            for i in range(self.num_class):
                #print("valid: ", pts[i, 2])
                #print(pts[i])
                if pts[i, 2] > 0: # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                    target15[i] = generate_heatmap(target15[i], pts[i], self.gk15)
                    target11[i] = generate_heatmap(target11[i], pts[i], self.gk11)
                    target9[i] = generate_heatmap(target9[i], pts[i], self.gk9)
                    target7[i] = generate_heatmap(target7[i], pts[i], self.gk7)
                    #print(target7[i])
            if self.vis and self.is_train:
                heatmap = np.zeros((self.out_res[0], self.out_res[1]))
                for i in range(self.num_class):
                    for row in range(self.out_res[0]):
                        for col in range(self.out_res[1]):
                            heatmap[row][col] += target7[i][row][col]
                cv2.imshow("", heatmap)
                cv2.waitKey(0)
                #cv2.destroyAllWindows()

            targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
            valid = pts[:, 2]

        if self.is_train:
            #print("-----------------------------")
            return img, targets, valid
        else:
            #print("*****************************")
            return img, image_name, os.path.join(self.img_path, image_name)

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(KPloader(cfg), batch_size=cfg.batch_size)
    for i, (img, targets, valid) in enumerate(train_loader): 
        print(i, img, targets, valid)

