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
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import re
from scipy import signal # imports to make spectrogram images

CWD = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(CWD, 'data')
SPECTROGRAM_ROOT = os.path.join(CWD, 'spectrogram-images')
PATTERN = os.path.join(ROOT_PATH, '**', '*.bdf')
TEST_FILE = './data/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf'
NONPD_PATH = r'.*sub-hc\d{1,2}.*'
PD_PATH = r'.*sub-pd\d{1,2}.*'

INTERVAL = 2560
FREQUENCY = 512
M = 256
MAX_AMP = 2
CHANNEL_COUNT = 40

def handle_arguments():
    """
    Function used to set arguments that can be passed to the script
    :return args: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-c', '--class', dest='classes', choices=['PD', 'NONPD', 'ALL'],
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

def clean_and_create(path):
    """
    Function used to clear out a directory and create a new one
    :param path: directory path to clean and create
    """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

    os.makedirs(path)

def load_data(filename):
    """
    Function used to load EEG data
    :param filename: filename of .bdf file we want to read in
    :return data: the raw date of given filename
    """
    raw_data = mne.io.read_raw_bdf(filename)

    """
        Informative parameters

        raw_data.info["sfreq"]
        raw_data.info["nchan"]
        raw_data.ch_names
    """
    return raw_data

def get_patient_path(filename, class_root, label):
    """
    Function used to get the patients location to place spectrogram images
    """
    filename_dir = filename.split(os.path.sep)[-4:-2]
    patient_path = os.path.join(class_root, '/'.join(filename_dir))

    return patient_path

def iterate_eeg_data(**kwargs):
    """
    Function used to iterate through EEG data and generate spectrogram images
    """
    counter = 0
    raw_data = kwargs["data"].copy()
    kwargs["data"].load_data()
    data = kwargs["data"].to_data_frame()
    channel_names = raw_data.ch_names
    fs = raw_data.info["sfreq"]

    for channel in channel_names:
        # Create channel output directory and iterate through all channels
        channel_path = os.path.join(kwargs["output_dir"], channel)
        clean_and_create(channel_path)

        channel_data = data[channel].values
        size = len(channel_data)

        # TODO: Need to determine dynamic way to iterate data
        i = 0
        j = 2048
        move = 1024
        while j < size:
            sub_channel_data = channel_data[i:j]
            output_file = os.path.join(channel_path, str(counter))

            generate_stft_from_data(fs=fs, m=M, sub_data=sub_channel_data, max_amp=MAX_AMP, output_filepath=output_file)

            i += move
            j += move
            counter += 1


def handle_create_spectrograms(**kwargs):
    """
    Function used to handle creating spectrogram images
    :param state: variable denoting what spectrograms to create
    :param root_path: root path to output specogram images
    :return: True if successful, False otherwise
    """
    class_list = []

    if (kwargs["state"] == 'ALL'):
        class_list = ['NONPD', 'PD']
    else:
        class_list = [kwargs["state"]]

    # need to check if output directories exist, create new
    clean_and_create(kwargs["root_path"])

    for curr_class in class_list:
        # Make directory for class (e.g., PD, NONPD)
        class_root = os.path.join(kwargs["root_path"], curr_class)
        clean_and_create(class_root)

        # need to read every patient EEG reading
        for filename in kwargs["all_files"][curr_class]:
            """
                1. Need to load in the data
            """
            data = load_data(filename)

            """
                2. Create output dir for patient data
            """
            patient_path = get_patient_path(filename, class_root, curr_class)
            clean_and_create(patient_path)

            """
                3. Create spectrogram images from the data
            """
            iterate_eeg_data(data=data, output_dir=patient_path)

def generate_stft_from_data(**kwargs):
    """
    Function use to generate Fast-Time Fourier Transform (stft) from data
    """
    noverlap = math.floor(kwargs["m"] * 0.9)
    nfft = kwargs["m"]

    f, t, Zxx = signal.stft(kwargs["sub_data"], kwargs["fs"], window='blackman',
                            nperseg=kwargs["m"], noverlap=noverlap, nfft=nfft)

    try:
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=kwargs["max_amp"])
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        # plt.set_cmap('jet')
        # plt.title('STFT Magnitude')
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        plt.savefig(kwargs["output_filepath"], bbox_inches='tight', pad_inches=0, dpi=64)
        plt.clf()
    except FloatingPointError as e:
        print('Caught divide by 0 error: {0}'.format(kwargs["output_filepath"]))
        return

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

    """
        3. Handle generation of spectrogram images
    """
    all_files = {
        "NONPD": nonpd_list,
        "PD": pd_list
    }
    handle_create_spectrograms(state=args.classes, root_path=SPECTROGRAM_ROOT, all_files=all_files)

    exit(0)

if __name__ == '__main__':
    main()