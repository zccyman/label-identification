import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from torchviz import make_dot, make_dot_from_trace

from train_config import cfg

from config import cfg_hrnet
from config import update_config

from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from dataloader.KPloader import KPloader
from adabound import AdaBound

import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--gpus', default=[0, 3], type=list, metavar='N',
                        help='number of GPU to use (default: 1)')    
    parser.add_argument('--epochs', default=1280, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    args = parser.parse_args()

    return args

def net_vision(model, args):
    input = torch.rand(1, args.channels, args.height, args.width)
    g = make_dot(model(input), params=dict(model.named_parameters()))
    g.view()

def main():
    args = parse_args()
    update_config(cfg_hrnet, args)

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    #print('networks.'+ cfg_hrnet.MODEL.NAME+'.get_pose_net')
    model = eval('models.'+ cfg_hrnet.MODEL.NAME+'.get_pose_net')(
        cfg_hrnet, is_train=True
    )
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # show net
    args.channels = 3
    args.height = cfg.data_shape[0]
    args.width = cfg.data_shape[1]
    #net_vision(model, args)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    #torch.optim.Adam
    optimizer = AdaBound(model.parameters(),
                                lr = cfg.lr,
                                weight_decay=cfg.weight_decay)
    
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:        
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        #MscocoMulti(cfg),
        KPloader(cfg),
        batch_size=cfg.batch_size*len(args.gpus))
        #, shuffle=True,
        #num_workers=args.workers, pin_memory=True) 

    #for i, (img, targets, valid) in enumerate(train_loader): 
    #    print(i, img, targets, valid)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr)) 

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer)
        print('train_loss: ',train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

    logger.close()



def train(train_loader, model, criterion, optimizer):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    #for i, (inputs, targets, valid, meta) in enumerate(train_loader):
    for i, (inputs, targets, valid) in enumerate(train_loader):	
        input_var = inputs.cuda(non_blocking=True)

        target15, target11, target9, target7 = targets
        valid_var = valid.cuda(non_blocking=True)

        # compute output
        global_outputs = model(input_var)
        num_points = global_outputs.size(1)
        #global_outputs, refine_output = model(input_var)
        #score_map = refine_output.data.cpu()
        #batch_size = global_outputs.size(0)
        #num_joints = global_outputs.size(1)
        #print("batch_size: ", batch_size, "num_joints: ", num_joints)
        #print("w: ", global_outputs.size(2), "h: ", global_outputs.size(3))

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        # comput global loss and refine loss
        for global_output, label in zip(global_outputs, targets):
            #print("num_points: ", num_points)
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion(global_output, global_label.cuda(non_blocking=True)) / 2.0
            loss += global_loss
            global_loss_record += global_loss.item()

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i%100==0 and i!=0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                .format(i, loss.data.item(), global_loss_record, 
                    refine_loss_record, losses.avg)) 

    return losses.avg

if __name__ == '__main__':
    main()
