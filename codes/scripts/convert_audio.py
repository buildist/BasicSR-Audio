import os, sys
from multiprocessing import Pool
import soundfile as sf
import numpy as np
from scipy import signal

def main():
    """A multi-thread tool for converting RGB images to gary/Y images."""

    input_folder = "C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/test"
    save_folder = "C:/Users/Jacob/Desktop/SuperResolution/BasicSR-Audio/data/test"
    n_thread = 8  # thread number

    audio_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]
        audio_list.extend(path)

    pool = Pool(n_thread)
    for path in audio_list:
        print(path)
        r = pool.apply_async(worker, args=(path, save_folder))
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder):
    print(path)
    audio_name = os.path.basename(path)
    audio, rate = sf.read(path, dtype="float32", always_2d=True)

    _, __, freq_left = signal.stft(audio[0:4194304,0], 10e3, nperseg=1000)
    _, __, freq_right = signal.stft(audio[0:4194304,1], 10e3, nperseg=1000)
    freq_left_amplitude = np.real(freq_left)
    freq_left_phase = np.imag(freq_left)
    freq_right_amplitude = np.real(freq_right)
    freq_right_phase = np.imag(freq_right)
    freq = np.dstack((freq_left_amplitude, freq_left_phase, freq_right_amplitude, freq_right_phase))
    np.save(os.path.join(save_folder, audio_name), freq)

    # sf.write(os.path.join(save_folder, audio_name), audio_rec, 44100, format='WAV')

def test():
    audio, rate = sf.read("test.ogg", dtype="float32", always_2d=True)

    _, __, freq_left = signal.stft(audio[:,0], 10e3, nperseg=1000)
    _, __, freq_right = signal.stft(audio[:,1], 10e3, nperseg=1000)
    freq_left_amplitude = np.real(freq_left)
    freq_left_phase = np.imag(freq_left)
    freq_right_amplitude = np.real(freq_right)
    freq_right_phase = np.imag(freq_right)
    freq = np.dstack((freq_left_amplitude, freq_left_phase, freq_right_amplitude, freq_right_phase))
    
    freq_left = freq[:,:,0] + 1j * freq[:,:,1]
    freq_right = freq[:,:,2] + 1j * freq[:,:,3]
    _, rec_left = signal.istft(freq_left, 10e3)
    _, rec_right = signal.istft(freq_right, 10e3)
    audio_rec = np.vstack((rec_left, rec_right)).T
    sf.write("test2.wav", audio_rec, 44100, format='WAV', subtype="PCM_24")

if __name__ == '__main__':
    main()
    #test()