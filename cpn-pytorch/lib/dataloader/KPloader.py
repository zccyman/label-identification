import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
from utils.create_csv import create_csv

class KPloader(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.vis = self.cfg.vis
        self.is_train = is_train
        self.num_class = cfg.num_class
        
        self.featuremap_scale = cfg.featuremap_scale
        self.out_res = cfg.output_shape
        self.data_shape = cfg.data_shape
        
        self.pixel_means = cfg.pixel_means
        
        self.img_path = cfg.img_path
        
        #create csv
        create_csv(self.img_path, self.cfg.csv_file_path, self.num_class, self.is_train)
        
        self.csv_file = pd.read_csv(self.cfg.csv_file_path)
        #self.csv_file = self.csv_file.rename(columns=lambda x: x.replace('[','').replace(']','').replace('"',''))
        #print(self.csv_file)
        
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        image_name = self.csv_file["image_name"][index]
        
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
            #keypoint scale
            keypoint[:, 0] //= float(image.shape[1] / self.data_shape[1])  #x
            keypoint[:, 1] //= float(image.shape[0] / self.data_shape[0])  #y

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
                    target15[i] = generate_heatmap(target15[i], pts[i], self.cfg.gk15)
                    target11[i] = generate_heatmap(target11[i], pts[i], self.cfg.gk11)
                    target9[i] = generate_heatmap(target9[i], pts[i], self.cfg.gk9)
                    target7[i] = generate_heatmap(target7[i], pts[i], self.cfg.gk7)
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
            return img, targets, valid
        else:
            return img, image

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(KPloader(cfg), batch_size=cfg.batch_size)
    for i, (img, targets, valid) in enumerate(train_loader): 
        print(i, img, targets, valid)

