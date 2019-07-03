import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT audio pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None

        self.paths_GT, self.sizes_GT = util.get_audio_paths(opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_audio_paths(opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of audio files - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        GT_size = self.opt['GT_size']

        # get GT audio
        GT_path = self.paths_GT[index]
        audio_GT = util.read_audio(GT_path)

        # get LQ audio
        LQ_path = self.paths_LQ[index]
        resolution = None
        audio_LQ = util.read_audio(LQ_path)

        if self.opt['phase'] == 'train':

            LQ_size = GT_size

            # randomly crop
            start = random.randint(0, audio_GT.size - GT_size)
            stop = start + GT_size
            audio_GT = audio_GT[start:stop]
            audio_LQ = audio_LQ[start:stop]

            # augmentation - reverse
            audio_LQ, audio_GT = util.augment([audio_LQ, audio_GT], self.opt['use_reverse'])
        audio_GT = np.ascontiguousarray(np.transpose(audio_GT))
        audio_GT = torch.from_numpy(audio_GT).float()
        audio_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(audio_LQ))).float()

        return {'LQ': audio_LQ, 'GT': audio_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)