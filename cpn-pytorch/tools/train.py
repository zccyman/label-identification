import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from torchviz import make_dot, make_dot_from_trace

from train_config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network 
from dataloader.KPloader import KPloader
from adabound import AdaBound

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--gpus', default=[0, 3], type=list, metavar='N',
                        help='number of GPU to use (default: 1)')    
    parser.add_argument('--epochs', default=1280, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='checkpoint/epoch1checkpoint.ckpt', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    args = parser.parse_args()

    return args

def net_vision(model, args):
    input = torch.rand(1, args.channels, args.height, args.width)
    g = make_dot(model(input), params=dict(model.named_parameters()))
    g.view()

def main():
    args = parse_args()

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network.__dict__[cfg.model](cfg.channel_settings, cfg.output_shape, cfg.num_class, pretrained = True)
    
    # show net
    args.channels = 3
    args.height = cfg.data_shape[0]
    args.width = cfg.data_shape[1]
    #net_vision(model, args)

    if 1:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            lr = checkpoint['lr']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else: 
        lr = cfg.lr
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])
     
    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda() # for Global loss
    criterion2 = torch.nn.MSELoss(reduce=False).cuda() # for refine loss
    
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
                                    
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        #MscocoMulti(cfg),
        KPloader(cfg), batch_size=cfg.batch_size*len(args.gpus))
        #, shuffle=True,
        #num_workers=args.workers, pin_memory=True) 
    
    #torch.optim.Adam
    optimizer = AdaBound(model.parameters(),
                        lr=lr,
                        weight_decay=cfg.weight_decay)
                                
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr)) 

        # train for one epoch
        train_loss = train(train_loader, model, [criterion1, criterion2], optimizer)
        print('train_loss: ',train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        #save_model({
        #    'epoch': epoch + 1,
        #    'state_dict': model.state_dict(),
        #    'optimizer' : optimizer.state_dict(),
        #}, checkpoint=args.checkpoint)

        state_dict = model.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'lr': lr,
            }, os.path.join(args.checkpoint, "epoch" + str(epoch + 1) + "checkpoint.ckpt"))
        print("=> Save model done! the path: ", \
              os.path.join(args.checkpoint, "epoch" + str(epoch + 1) + "checkpoint.ckpt"))
        
    logger.close()

def train(train_loader, model, criterions, optimizer):
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
    criterion1, criterion2 = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    #for i, (inputs, targets, valid, meta) in enumerate(train_loader):
    for i, (inputs, targets, valid) in enumerate(train_loader):
        input_var = inputs.cuda(non_blocking=True)

        target15, target11, target9, target7 = targets
        refine_target_var = target7.cuda(non_blocking=True)
        valid_var = valid.cuda(non_blocking=True)

        # compute output
        global_outputs, refine_output = model(input_var)
        score_map = refine_output.data.cpu()

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        # comput global loss and refine loss
        #i_loss = 0
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            #print("num_points: ", num_points)
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            #print(global_label, i_loss)
            #i_loss = i_loss + 1
            global_loss = criterion1(global_output, global_label.cuda(non_blocking=True)) / 2.0
            loss += global_loss
            global_loss_record += global_loss.item()
            
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 40)
        loss += refine_loss
        refine_loss_record = refine_loss.item()

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i%20==0 and i!=0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                .format(i, loss.data.item(), global_loss_record, 
                    refine_loss_record, losses.avg)) 

    return losses.avg

if __name__ == '__main__':
    main()
