import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LRDataset(data.Dataset):
    '''Read LR audio only in the test phase.'''

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.paths_LQ = None
 
        self.paths_LQ, _ = util.get_audio_paths(opt['dataroot_LQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def __getitem__(self, index):
        LQ_path = self.paths_LQ[index]
        audio_LQ = util.read_audio(LQ_path)
        audio_LQ = torch.from_numpy(np.transpose(audio_LQ)).float()

        return {'LQ': audio_LQ, 'LQ_path': LQ_path}

    def __len__(self):
        return len(self.paths_LQ)
