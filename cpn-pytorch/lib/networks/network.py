from .resnet import *
import torch.nn as nn
import torch
from .globalNet import globalNet
from .refineNet import refineNet

__all__ = ['CPN18', 'CPN50', 'CPN101', 'CPN152']

class CPN(nn.Module):
    def __init__(self, resnet, channel_settings, output_shape, num_class, pretrained=True):
        super(CPN, self).__init__()
        self.channel_settings_gloabl = channel_settings[1]
        self.resnet = resnet
        self.global_net = globalNet(self.channel_settings_gloabl, output_shape, num_class)
        self.channel_settings_refine = channel_settings[1] #= [2048, 1024, 512, 256]
        self.refine_net = refineNet(self.channel_settings_refine[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        global_fms, global_outs = self.global_net(res_out)
        refine_out = self.refine_net(global_fms)

        return global_outs, refine_out

def CPN18(channel_settings, out_size,num_class,pretrained=True):
    res18 = resnet18(pretrained=pretrained)
    model = CPN(res18, channel_settings=channel_settings, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN50(channel_settings, out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, channel_settings=channel_settings, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN101(channel_settings, out_size,num_class,pretrained=True):
    res101 = resnet101(pretrained=pretrained)
    model = CPN(res101, channel_settings=channel_settings, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

def CPN152(channel_settings, out_size,num_class,pretrained=True):
    res152 = resnet152(pretrained=pretrained)
    model = CPN(res152, channel_settings=channel_settings, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model