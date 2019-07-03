import numpy as np
import math
from data import create_dataloader, create_dataset
from utils import util
import soundfile as sf

if __name__ == '__main__':
    opt = {}

    opt['name'] = 'DIV2K800'
    opt['dataroot_GT'] = 'C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/test/GT'
    opt['dataroot_LQ'] = 'C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/test/LQ'
    opt['mode'] = 'LQGT'
    opt['phase'] = 'train'  # 'train' | 'val'
    opt['use_shuffle'] = True
    opt['n_workers'] = 8
    opt['batch_size'] = 16
    opt['GT_size'] = 131072
    opt['use_reverse'] = False
    opt['gpu_ids'] = [0]
    opt["dist"] = False

    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt, opt, None)

    for i, data in enumerate(train_loader):
        print(data['GT'].shape)
        GT = np.transpose(data['GT'][0].numpy()) * 2. - 1.
        LQ = np.transpose(data['LQ'][0].numpy()) * 2. - 1.
        sf.write('GT.wav', GT, 44100, format='WAV', subtype="PCM_24")
        sf.write('LQ.wav', LQ, 44100, format='WAV', subtype='PCM_24')
