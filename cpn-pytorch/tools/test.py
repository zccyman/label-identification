import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import math
import numpy as np

from test_config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network 
from dataloader.KPloader import KPloader

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--gpus', default=[0], type=list, metavar='N',
                        help='number of GPU to use (default: 1)')     
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=1, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='epoch85checkpoint', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--results', default='../data/351/cpn-results', type=str,
                        help='path to save save result (default: result)')
    
    args = parser.parse_args()

    return args
    
def main():
    args = parse_args()

    # create model
    model = network.__dict__[cfg.model](cfg.channel_settings, cfg.output_shape, cfg.num_class, pretrained = False)
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    test_loader = torch.utils.data.DataLoader(
        KPloader(cfg), batch_size=args.batch*len(args.gpus)
        ) 

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test + '.ckpt')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    
    # change to evaluation mode
    model.eval()
           
    print('testing...')
   
    for i, (inputs, image_name, image_path) in enumerate(test_loader):
        print(i, image_path[0])
        #input("")
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            
            # compute output
            global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()
            #print(score_map.shape[0], score_map.shape[1], score_map.shape[2], score_map.shape[3])
            
            if 1:
                image = cv2.imread(image_path[0], 1)
                
                if 2 == image.ndim:
                    #print("GRAY")
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif 3 == image.ndim:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
                #print(image.shape[0], image.shape[1], image.shape[2], image.shape[3]) # h w c
                image_h = image.shape[0]
                image_w = image.shape[1]
                
                def tensor_to_np(tensor):
                    img = tensor.byte()
                    img = img.cpu().numpy().squeeze(0)#.transpose((1, 2, 0))
                    return img

                #image = tensor_to_np(image)
                
                scale_x, scale_y = (float(image_w / score_map.shape[3]), float(image_h / score_map.shape[2]))
                #print(scale_x, scale_y)

                #print(score_map.shape[2], score_map.shape[3]) #h w

                heatmap = np.zeros((image_h, image_w, 3))
                for i in range(score_map.shape[1]):
                    max_score = 0.0
                    _row = 0
                    _col = 0
                    for row in range(score_map.shape[2]):
                        for col in range(score_map.shape[3]):
                            if score_map[0][i][row][col] > max_score:
                                _row = row 
                                _col = col 
                                max_score = score_map[0][i][row][col]
                    #heatmap[math.floor(_row * scale_y)][math.floor(_col * scale_x)] = (0, 0, 255)
                    #image[math.floor(_row * scale_y), math.floor(_col * scale_x)] = (0, 0, 255)
                    cv2.circle(heatmap, (math.floor(_col * scale_x), math.floor(_row * scale_y)), 5, (0, 0, 255), -1) #x y
                    cv2.circle(image, (math.floor(_col * scale_x), math.floor(_row * scale_y)), 5, (0, 0, 255), -1) #x y

                if cfg.vis:
                    cv2.imshow("origin", image)
                    cv2.imshow("heatmap", heatmap)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    save_path = args.results
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, image_name[0]), image)
                    #input("")

def demo():
    args = parse_args()

    # create model
    model = network.__dict__[cfg.model](cfg.channel_settings, cfg.output_shape, cfg.num_class, pretrained = False)
    model = torch.nn.DataParallel(model).cuda()

    # load trainning weights
    checkpoint_file = os.path.join(args.checkpoint, args.test + '.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    
    # change to evaluation mode
    model.eval()
           
    print('testing...')
   
    for i, (inputs, image_h, image_w, image_name) in enumerate(test_loader):
        print(i, image_h, image_w, image_name)
        #input("")
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            
            # compute output
            global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()
            #print(score_map.shape[0], score_map.shape[1], score_map.shape[2], score_map.shape[3])
            
            if 1:
                #print(image.shape[0], image.shape[1], image.shape[2], image.shape[3]) # h w c
                #image_h = image.shape[1]
                #image_w = image.shape[2]
                
                def tensor_to_np(tensor):
                    img = tensor.byte()
                    img = img.cpu().numpy().squeeze(0)#.transpose((1, 2, 0))
                    return img

                #image = tensor_to_np(image)
                
                #scale_x, scale_y = (float(image_w / score_map.shape[3]), float(image_h / score_map.shape[2]))
                #print(scale_x, scale_y)

                #print(score_map.shape[2], score_map.shape[3]) #h w

                heatmap = np.zeros((image_h, image_w, 3))
                for i in range(score_map.shape[1]):
                    max_score = 0.0
                    _row = 0
                    _col = 0
                    for row in range(score_map.shape[2]):
                        for col in range(score_map.shape[3]):
                            if score_map[0][i][row][col] > max_score:
                                _row = row 
                                _col = col 
                                max_score = score_map[0][i][row][col]
                    #heatmap[math.floor(_row * scale_y)][math.floor(_col * scale_x)] = (0, 0, 255)
                    #image[math.floor(_row * scale_y), math.floor(_col * scale_x)] = (0, 0, 255)
                    cv2.circle(heatmap, (math.floor(_col * scale_x), math.floor(_row * scale_y)), 2, (0, 0, 255), -1) #x y
                    #cv2.circle(image, (math.floor(_col * scale_x), math.floor(_row * scale_y)), 2, (0, 0, 255), -1) #x y

                if cfg.vis:
                    cv2.imshow("origin", image)
                    cv2.imshow("heatmap", heatmap)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    save_path = args.result
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, image_name), heatmap)
                    input("")

if __name__ == '__main__':
    main()
