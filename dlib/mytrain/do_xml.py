#coding:utf-8

import os
import xml.dom.minidom
import random

import cv2
import numpy as np
import shutil
import math

global xlabel, ylabel

def get_mean(files_in, image_w = 1280, image_h = 1024, key_nums = 41):
    xlabel = np.zeros(key_nums)
    ylabel = np.zeros(key_nums)

    sample_num = 0
    for im in os.listdir(files_in):
        if postfix in im:
            im = im.split("." + postfix)[0]
            if im == 'rec':
                continue

            sample_num = sample_num + 1

            pts_name = files_in + im + '.pts'
            fpts = open(pts_name,'r')
            fptsLines = fpts.readlines()
            
            print(pts_name, len(fptsLines))
            for n in range(3, len(fptsLines) - 1):
                fpts_line = fptsLines[n].strip().split(' ')
                #print(float(fpts_line[0]), float(fpts_line[1]))
                xlabel[n - 3] += (float(fpts_line[0]) / image_w)
                ylabel[n - 3] += (float(fpts_line[1]) / image_h)
            fpts.close()

    for n in range(0, key_nums):
        xlabel[n] /= sample_num
        ylabel[n] /= sample_num
        
    for n in range(0, key_nums):
        xlabel[n] = int(xlabel[n] * image_w)
        ylabel[n] = int(ylabel[n] * image_h)
    
    #with open(files_in + "anchor_x.txt","a") as f:
    #    for i in range(0, len(xlabel)):    
    #        f.write(str(int(xlabel[i] * 2)) + ", ")
    #f.close()

    #with open(files_in + "anchor_y.txt","a") as f:
    #    for i in range(0, len(ylabel)):    
    #        f.write(str(int(ylabel[i] * 2)) + ", ")
    #f.close()

    return xlabel, ylabel

#制作xml文件（后1%做测试）    
def GenerateXml(scale_size, filess, start, end, mode = 0):
    nm = 0
    filesss = filess[start:end]
    impl = xml.dom.minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'images', None)
    root = dom.documentElement
    for im in filesss:
        nm = nm + 1
        for randr in range(1):#disturb num = 10
            employee = dom.createElement('image')
            employee.setAttribute("file",im)
            root.appendChild(employee)
            
            f=open(files_in + 'rec.txt', 'r')
            rec = f.readlines()
            rec = rec[start:end]
            #print(nm)
            wh = rec[nm-1].rstrip().split(' ')
            #print(wh)

            nameE=dom.createElement('box')
            rdm_y = int(wh[0]) #+ random.randint(-20, 20)
            nameE.setAttribute("top",str(rdm_y)) #disturb y
            employee.appendChild(nameE)
            rdm_x = int(wh[1]) #+ random.randint(-20, 20)
            nameE.setAttribute("left",str(rdm_x) ) #disturb x
            employee.appendChild(nameE)
            
            nameE.setAttribute("width",str(wh[2]))
            employee.appendChild(nameE)
            rdm_h = int(wh[3]) #+ random.randint(-20, 20)#disturb height
            nameE.setAttribute("height",str(rdm_h))
            employee.appendChild(nameE)
            f.close()
            
            ptsfile =  files_out + im.rstrip('.bmp') + '.txt'
            print(ptsfile)

            f= open(ptsfile, 'r')
            line = f.readline()
            num_pts = f.readline().rstrip()[-2:]
            line = f.readline()
            for i in range(int(num_pts)):
                xy = f.readline().rstrip().split(' ')
                nameP=dom.createElement('part')
                if i < 10:
                    i0 = '0' + str(i)
                    nameP.setAttribute("name",i0)
                    nameE.appendChild(nameP)
                else:
                    nameP.setAttribute("name",str(i))
                    nameE.appendChild(nameP)
                nameP.setAttribute("x",str(int(float(xy[0]) / scale_size)))
                nameE.appendChild(nameP)
                nameP.setAttribute("y",str(int(float(xy[1]) / scale_size)))
                nameE.appendChild(nameP)
            f.close()
    if mode == 0:
         f= open(files_out + 'training_with_face_landmarks.xml', 'a')
         dom.writexml(f, addindent='  ', newl='\n', encoding='ISO-8859-1')
         f.close()
    else:
         f= open(files_out + 'testing_with_face_landmarks.xml', 'a')
         dom.writexml(f, addindent='  ', newl='\n', encoding='ISO-8859-1')
         f.close()
    print(nm)
    del employee

    return 0

def create_rect(files_in, files_out, scale_size):
    f = open(files_in + 'rec.txt','w')
    for im in os.listdir(files_in):
        if postfix in im:
            im = im.split("." + postfix)[0]
            if im == 'rec':
                continue
 
            print(files_in + im)
            #print(im)
            #input("")
            
            if os.path.exists(files_in + im + ".jpg"):
                print(files_in + im + ".jpg")
                image = cv2.imread(files_in + im + ".jpg", 0)
            elif os.path.exists(files_in + im + ".bmp"):
                print(files_in + im + ".bmp")
                image = cv2.imread(files_in + im + ".bmp", 0)

            if not os.path.exists(files_out):
                os.makedirs(files_out)

            shutil.copyfile(files_in + im + '.' + postfix, files_out + "scale_" + im + ".txt")

            image_width= int(image.shape[1] / scale_size)
            image_height= int(image.shape[0] / scale_size)
            image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            if (image.ndim > 2):
               image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            start_x = int(min(xlabel))
            start_y = int(min(ylabel))
            end_x = int(max(xlabel))
            end_y = int(max(ylabel))

            cv2.imwrite(files_out + "scale_" + im + ".bmp", image)
            print(start_x, " ", start_y, " ", end_x, " ", end_y, " ")
            #cv2.rectangle(image,(start_x,start_y),(end_x ,end_y),(255,255,255),3)
            #cv2.imshow('image',image)
            #cv2.waitKey(0)

            f.write(str(start_y) + ' ' + str(start_x) +  ' ' + str(end_x - start_x) + ' ' + str(end_y - start_y) + ' ' + str(image_width) + ' ' + str(image_height) + '\n')

    f.close()

    return 0

def create_xml(files_in, files_out, scale_size):
    files = os.listdir(files_out)
    filess = []
    for im in files:
        if 'bmp' in im:
            filess.append(im)

    len_all = len(filess)
    print(len_all)
    len_train = len_all
    GenerateXml(scale_size, filess, 0, len_train)

    return 0

def make_xml_perfect(files_in, files_out):
    #xml文件完善
    f= open(files_out + 'training_with_face_landmarks.xml', 'r')
    c = f.readlines()
    f.close()
    a = ["<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n",'<dataset>\n','<name>Training faces</name>\n','<comment>These are images from the PASCAL VOC 2011 dataset.\n',"   The face landmarks are from dlib's shape_predictor_68_face_landmarks.dat\n",'   landmarking model.  The model uses the 68 landmark scheme used by the\n','iBUG\n','   300-W dataset.\n','</comment>\n','</dataset>\n']
    c_a = list(c[0]) + a[0:9] + c[1:] + list(a[-1])
    f= open(files_out + 'training_with_face_landmarks.xml', 'w')
    for i in c_a:
        f.write(i)
    f.close()

    return 0

def rename(files_in, files_out):
    postfix = "jpg"
    for im in os.listdir(files_in):
        if postfix in im:
            im = im.split("." + postfix)[0]
            print(files_in + im + "." + postfix)
            print(files_out + im[0:-4] + ".jpg")
            shutil.copyfile(files_in + im + "." + postfix, files_out + im[0:-4] + ".jpg")
            shutil.copyfile(files_in + im + ".pts", files_out + im[0:-4] + ".pts")

    return 0

def augment(files_in, files_out, key_nums):
    postfix = "pts"
    for im in os.listdir(files_in):
        if postfix in im:
            im = im.split("." + postfix)[0]
            
            image = cv2.imread(files_in + im + ".jpg", 0)
            print(files_in + im + ".jpg")

            xlabel = np.zeros(key_nums)
            ylabel = np.zeros(key_nums)
            pts_name = files_in + im + '.pts'
            fpts = open(pts_name,'r')
            fptsLines = fpts.readlines()
            #print("line_num:", len(fptsLines))
            #input("")
            if (key_nums + 4) != len(fptsLines):
                continue
            
            shutil.copyfile(files_in + im + ".jpg", files_out + im + ".jpg")
            shutil.copyfile(files_in + im + ".pts", files_out + im + ".pts")

            for n in range(3, len(fptsLines) - 1):
                fpts_line = fptsLines[n].strip().split(' ')
                #print(float(fpts_line[0]), float(fpts_line[1]))
                xlabel[n - 3] += float(fpts_line[0])
                ylabel[n - 3] += float(fpts_line[1])
            fpts.close()

            #augment image.shape[0] = rows
            for a in range(0, 9):
                if 0 == a: #h
                    map_talbel = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 20, 19, 18, 17, 16]
                    pts_name = files_out + im + '_h.pts'
                    with open(pts_name,"a") as f:
                        f.write("version:1" + "\n")
                        f.write("n_points :" + str(key_nums) +  "\n")
                        f.write("{" + "\n")
                        for i in range(0, len(ylabel)):
                            tx = image.shape[1] - xlabel[map_talbel[i] - 1]
                            ty = ylabel[map_talbel[i] - 1]    
                            f.write(str(tx) + " " + str(ty) + "\n")
                        f.write("}" + "\n")
                    f.close()

                    image_hflip = cv2.flip(image, 1)
                    cv2.imwrite(files_out + im + '_h.jpg', image_hflip)
                elif 1 == a:#v
                    map_talbel = [5, 4, 3, 2, 1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
                    pts_name = files_out + im + '_v.pts'
                    with open(pts_name,"a") as f:
                        f.write("version:1" + "\n")
                        f.write("n_points :" + str(key_nums) + "\n")
                        f.write("{" + "\n")
                        for i in range(0, len(ylabel)):
                            tx = xlabel[map_talbel[i] - 1]
                            ty = image.shape[0] - ylabel[map_talbel[i] - 1]    
                            f.write(str(tx) + " " + str(ty) + "\n")
                        f.write("}" + "\n")
                    f.close()

                    image_hflip = cv2.flip(image, 0)
                    cv2.imwrite(files_out + im + '_v.jpg', image_hflip)
                elif 2 == a:#hv
                    map_talbel = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    pts_name = files_out + im + '_hv.pts'
                    with open(pts_name,"a") as f:
                        f.write("version:1" + "\n")
                        f.write("n_points :" + str(key_nums) + "\n")
                        f.write("{" + "\n")
                        for i in range(0, len(ylabel)):
                            tx = image.shape[1] - xlabel[map_talbel[i] - 1]
                            ty = image.shape[0] - ylabel[map_talbel[i] - 1]    
                            f.write(str(tx) + " " + str(ty) + "\n")
                        f.write("}" + "\n")
                    f.close()

                    image_hflip = cv2.flip(image, -1)
                    cv2.imwrite(files_out + im + '_hv.jpg', image_hflip)
                else:#rotation
                    center_x = image.shape[1] >> 1
                    center_y = image.shape[0] >> 1
                    def rotate_point(x0, y0, center_x, center_y, theta):
                        pi = 3.1415926535897932384626433832795
                        theta = theta * pi / 180.0
                        x = (x0 - center_x) * math.cos(theta) - (y0 - center_y) * math.sin(theta) + center_x
                        y = (x0 - center_x) * math.sin(theta) + (y0 - center_y) * math.cos(theta) + center_y
                        
                        x = math.floor(x)
                        y = math.floor(y)

                        if x > image.shape[1]: x = image.shape[1] - 1
                        elif x < 0: x = 0
                        
                        if y > image.shape[0]: y = image.shape[0] - 1
                        elif y < 0: y = 0

                        return x, y

                    angle = [-15, -10, -5, 5, 10, 15]
                    map_talbel = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                    pts_name = files_out + im + '_' + str(angle[a - 3]) + '_rotate.pts'
                    with open(pts_name,"a") as f:
                        f.write("version:1" + "\n")
                        f.write("n_points :" + str(key_nums) + "\n")
                        f.write("{" + "\n")
                        for i in range(0, len(ylabel)):
                            #tx = image.shape[1] - xlabel[map_talbel[i] - 1]
                            #ty = image.shape[0] - ylabel[map_talbel[i] - 1]
                            tx, ty = rotate_point(xlabel[map_talbel[i] - 1],  \
                            ylabel[map_talbel[i] - 1], \
                            center_x, center_y, \
                            -angle[a - 3])

                            f.write(str(tx) + " " + str(ty) + "\n")
                        f.write("}" + "\n")
                    f.close()

                    M = cv2.getRotationMatrix2D((center_x, center_y), angle[a - 3], 1.0)
                    image_hflip = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                    cv2.imwrite(files_out + im + '_' + str(angle[a - 3]) + '_rotate.jpg', image_hflip)
                
    return 0

def if_no_exist_path_and_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return 0

if __name__ == "__main__":
    #files_in = "F:\\workspace\\KeyPoint\\TrainData\\DTY\\dtylabel\\origin\\"
    #files_out = "F:\\workspace\\KeyPoint\\TrainData\\DTY\\dtylabel\\augment\\"
    files_in = "data/origin/"
    files_out = "data/augment/"
    if_no_exist_path_and_make_path(files_in)
    if_no_exist_path_and_make_path(files_out)

    #augment
    #rename(files_in, files_out)
    augment(files_in, files_out, 20)
    #input("")

    #files_in = "F:\\workspace\\KeyPoint\\TrainData\\DTY\\dtylabel\\augment\\"
    #files_out = "F:\\workspace\\KeyPoint\\TrainData\\DTY\\dtylabel\\train_data\\"
    files_in = "data/augment/"
    files_out = "data/train_data/"
    if_no_exist_path_and_make_path(files_in)
    if_no_exist_path_and_make_path(files_out)
    
    #rename(files_in, files_out)
    #input("")

    postfix = "pts"

    if os.path.exists(files_in + 'rec.txt'):
        os.remove(files_in + 'rec.txt')

    scale_size = 2 #放缩倍数

    #xlabel, ylabel = get_mean(files_in, 616, 436, 30)
    #print(int(min(xlabel) / scale_size), int(min(ylabel) / scale_size) , \
    # int(max(xlabel) / scale_size), int(max(ylabel) / scale_size))
    xlabel = [0, int(616 / scale_size)]
    ylabel = [0, int(436 / scale_size)]

    create_rect(files_in, files_out, scale_size)
    create_xml(files_in, files_out, scale_size)

    if os.path.exists(files_in + 'rec.txt'):
        os.remove(files_in + 'rec.txt')

    make_xml_perfect(files_in, files_out)
