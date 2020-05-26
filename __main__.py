#!/usr/bin/env python3
import mne
import glob
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy import signal # imports to make spectrogram images

ROOT_PATH = './data/'
TEST_FILE = './data/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf'

FREQUENCY = 512
M = 256
MAX_AMP = 2

def generate_stft_from_data(fs, m, max_amp, sub_data, sub_output_file):
    """
    Function use to generate Fast-Time Fourier Transform (stft) from data
    """
    pass

def generate_spectrogram_from_data(fs, m, data, output_filepath):
    """
    Function used to generate spectrogram images
    """
    overlap = math.floor(m * 0.9)

    f, t, Sxx = signal.spectrogram(data, fs, noverlap=overlap, window=signal.tukey(m, 0.25))

    try:
        plt.pcolormesh(t, f, np.log10(Sxx))
        plt.set_cmap('jet')
        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
        plt.clf()
    except FloatingPointError as e:
        print('Caught divide by 0 error: {0}'.format(output_filepath))
        return

def main():
    """
    Main Entrance of script
    """
    raw_data = mne.io.read_raw_bdf(TEST_FILE, preload=True)
    the_data = np.array(raw_data.get_data())
    print(raw_data.info)
    print(raw_data.annotations)
    print(the_data[0])
    data = raw_data.to_data_frame()
    sample_freq = raw_data.info["sfreq"]
    channel_num = raw_data.info["nchan"]
    channel_names = raw_data.ch_names
    print(data.head())
    print(data.info())

    curr_channel = data[channel_names[0]]
    values = curr_channel.values
    print('Successful read')

    # generate_spectrogram_from_data(FREQUENCY, M, values, 'testimage')
    fig = raw_data.plot(duration=3, n_channels=40, show=True, color='k')
    plt.show()


if __name__ == '__main__':
    main()