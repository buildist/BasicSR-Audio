import numpy as np
import math
from data import create_dataloader, create_dataset
from utils import util

if __name__ == '__main__':
    opt = {}

    opt['name'] = 'test'
    opt['dataroot_GT'] = 'C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/F/val'
    opt['dataroot_LQ'] = 'C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/F_mono/val'
    opt['mode'] = 'LQGT'
    opt['phase'] = 'train'  # 'train' | 'val'
    opt['use_shuffle'] = True
    opt['n_workers'] = 8
    opt['batch_size'] = 16
    opt['GT_size'] = 128
    opt['use_reverse'] = False
    opt['gpu_ids'] = [0]
    opt["dist"] = False

    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt, opt, None)

    for i, data in enumerate(train_loader):
        util.save_audio(data['GT'][0], 'GT.wav')
        util.save_audio(data['LQ'][0], 'LQ.wav')
