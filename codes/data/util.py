import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import soundfile as sf

####################
# Files & IO
####################

###################### get audio path list ######################
AUDIO_EXTENSIONS = ['.wav']


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def _get_paths_from_files(path):
    '''get file path list from folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_audio_file(fname):
                file_path = os.path.join(dirpath, fname)
                files.append(file_path)
    assert files, '{:s} has no valid audio file'.format(path)
    return files

def get_audio_paths(dataroot):
    '''get audio path list'''
    paths, sizes = None, None
    if dataroot is not None:
        paths = sorted(_get_paths_from_files(dataroot))
    return paths, sizes


###################### read audio ######################
def read_audio(path):
    '''read audio by scipy
    return: Numpy float32, [0,1]'''
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = (audio + 1)/2.
    return audio


####################
# audio processing
# process on numpy audio
####################


def augment(audio_list, reverse=True):
    # reverse
    reverse = reverse and random.random() < 0.5

    def _augment(audio):
        if reverse:
            audio = audio[:, ::-1]
        return audio

    return [_augment(audio) for audio in audio_list]
