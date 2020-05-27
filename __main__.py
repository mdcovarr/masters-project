#!/usr/bin/env python3
"""
    Main script to create spectrogram images

    Original Data Information
    --------------------------
    High Pass Filter: 0
    Low Pass Filter: 104
"""

import mne
import glob
import argparse
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import re
from scipy import signal # imports to make spectrogram images

CWD = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = 'data'
PATTERN = os.path.join(CWD, ROOT_PATH, '**', '*.bdf')
TEST_FILE = './data/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf'
NONPD_PATH = r'.*sub-hc\d{1,2}.*'
PD_PATH = r'.*sub-pd\d{1,2}.*'

INTERVAL = 2560
FREQUENCY = 512
M = 256
MAX_AMP = 104

def handle_arguments():
    """
    Function used to set arguments that can be passed to the script
    :return args: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-c', '--class', dest='class', choices=['PD', 'NONPD', 'ALL'],
            help='Flag used to determine what class type we want to cretae spectrogram images for')

    args = parser.parse_args()

    return args

def get_data_files(location):
    """
    Function used to get all files of interest in regards to EEG data
    :param location: full path location of where to search
    :return all_files: list of all files of interest
    """
    all_files = []

    all_files = glob.glob(location, recursive=True)

    return all_files

def generate_stft_from_data(fs, m, max_amp, sub_data, output_filepath):
    """
    Function use to generate Fast-Time Fourier Transform (stft) from data
    """
    noverlap = math.floor(m * 0.9)
    nfft = m

    f, t, Zxx = signal.stft(sub_data, fs, window='blackman', nperseg=m, noverlap=noverlap, nfft=nfft)

    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=max_amp)
    # plt.set_cmap('jet')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.savefig(output_filepath)

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
    args = handle_arguments()

    """
        1. Get list of data files EEG
    """
    data_files = get_data_files(PATTERN)

    """
        2. Separate PD VS NON PD Patients
    """
    nonpd_regex = re.compile(NONPD_PATH)
    pd_regex = re.compile(PD_PATH)

    nonpd_list = list(filter(nonpd_regex.match, data_files))
    pd_list = list(filter(pd_regex.match, data_files))

    # make sure we are able to split classes correctly
    if (len(data_files) != (len(nonpd_list) + len(pd_list))):
        print("Error: While separating PD and NON PD patients")
        exit(1)


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
    generate_stft_from_data(FREQUENCY, M, MAX_AMP, values[:2560], 'testimage')
    # fig = raw_data.plot(duration=3, n_channels=40, show=True, color='k')
    # plt.show()


if __name__ == '__main__':
    main()