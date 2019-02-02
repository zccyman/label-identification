#coding:utf-8

import os
import xml.dom.minidom
import random

import cv2
import numpy as np
import shutil
import math
import pandas as pd
    
def create_csv(files_in, files_out, key_nums):   
    f = open(files_out,'w') 
    f.write("image_name,gt_bbox,keypoints\n")
    for im in os.listdir(files_in):
        postfix = "pts"
        if postfix in im:
            im = im.split("." + postfix)[0]
            
            f.write(im + ".jpg,")
            for i in range(4):
                if 3 == i:
                    f.write(str(0) + ",")
                else:
                    f.write(str(0) + " ")
            
            pts_name = os.path.join(files_in, im + '.pts')
            fpts = open(pts_name,'r')
            fptsLines = fpts.readlines()            
            print(pts_name, len(fptsLines))
            for n in range(key_nums):
                fpts_line = fptsLines[n + 3].strip().split(' ')
                if (key_nums - 1) == n:
                    f.write(str(fpts_line[0]) + " " + str(fpts_line[1]) + " " + str(2))
                else:
                    f.write(str(fpts_line[0]) + " " + str(fpts_line[1]) + " " + str(2) + " ")
            fpts.close()
            
            f.write("\n")

    f.close()

if __name__ == "__main__":
    files_in = "data/train"
    files_out = "data/train.csv"
    key_nums = 20
    create_csv(files_in, files_out, key_nums)
