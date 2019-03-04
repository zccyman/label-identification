# PyTorch HRNET: Deep High-Resolution Representation Learning for Human Pose Estimation 

## Train Model with PTS DataFormat

### Prepare data
- ../hrnet-pytorch/data/origin/
    - *.jpg
    - *.pts

### Train Model

- pre-model: 
    - lib/models/pytorch/imagenet/[pose_hrnet_w48_256x192.pth](https://onedrive.live.com/?authkey=%21AEwfaSueYurmSRA&id=56B9F9C97F261712%2111776&cid=56B9F9C97F261712)

```
sh train.sh
```

### Test Model


## Acknowledgement

- [Deep High-Resolution Representation Learning for Human Pose Estimation(accepted to CVPR2019)](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

- [PyTorch CPN(Cascaded Pyramid Network)](https://github.com/GengDavid/pytorch-cpn)

