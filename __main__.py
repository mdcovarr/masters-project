#!/usr/bin/env python3
"""
    Main script to create spectrogram images

    Original Data Information
    --------------------------
    High Pass Filter: 0
    Low Pass Filter: 104


    Useful Information

    BDF file detected
    Setting channel info structure...
    Creating raw.info structure...
    <Info | 7 non-empty values
    bads: []
    ch_names: Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, ...

    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
    'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3',
    'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'


    chs: 40 EEG, 1 STIM
    custom_ref_applied: False
    highpass: 0.0 Hz
    lowpass: 104.0 Hz
    meas_date: 2011-02-04 11:50:44 UTC
    nchan: 41
    projs: []
    sfreq: 512.0 Hz
    --------------------------

    Images are created size: 496 x 369

    NONPD Patients 16
    PD Patients 15
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
from datautils import wavelet_transform
from datautils import data_loader
from datautils import stft

CWD = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = '' #os.path.join(CWD, 'data')
IMAGES_ROOT = ''

PATHS = [] # ['**/ses-hc/eeg/*.bdf', '**/ses-off/eeg/*.bdf', '**/ses-on/eeg/*.bdf']
EXCLUDED_CHANNELS = ['Status', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

def handle_arguments():
    """
    Function used to set arguments that can be passed to the script
    :return args: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-c', '--class', dest='classes', required=True, choices=['PD_OFF', 'PD_ON', 'NONPD', 'ALL'],
            help='Flag used to determine what class type we want to cretae spectrogram images for')
    parser.add_argument('-s', '--stft', dest='stft', action='store_true', default=False,
            help='Flag used to utilize the short-time fourier transform in data processing')
    parser.add_argument('-w', '--wave', dest='wavelet', action='store_true', default=False,
            help='Flag used to utilize wavelet transform in data processing')
    parser.add_argument('-i', '--input-dir', dest='input_dir', required=True,
            help='Flag used to determine the root input directory of the data')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
            help='Flag used to determine the root output path to place images')
    parser.add_argument('-a', '--ica', dest='ica', required=False
            help='Flag used to generate Independent Component Analysis of EEG data'
    )

    args = parser.parse_args()

    return args

def handle_morlet_wavelet_transform(**kwargs):
    """
    Function used to handle the creating of scalogram images
    using wavelet transform
    """
    wavelet_helper = wavelet_transform.WaveletTransform(**kwargs)
    wavelet_helper.generate_wavelet_transform()

def handle_stft(**kwargs):
    """
    Function used to handle the creating of spectrogram image
    using short-time fourier transform
    """
    stft_helper = stft.STFT(**kwargs)
    stft_helper.generate_stft_transform()

def main():
    """
    Main Entrance of script
    """
    args = handle_arguments()

    """
        Determine root output path for images
    """
    IMAGE_ROOT = os.path.join(CWD, args.output_dir)
    paths = ['**/ses-hc/eeg/*.bdf', '**/ses-off/eeg/*.bdf', '**/ses-on/eeg/*.bdf']

    for path in paths:
        PATHS.append(os.path.join(CWD, args.input_dir, path))

    """
        Get Data Helper to load files
    """
    data_helper = data_loader.DataLoader(paths=PATHS)
    data_helper.load_data_files(data_helper.paths)

    # Function used for the short time fourier transform
    if args.stft:
        handle_stft(state=args.classes, root_path=IMAGE_ROOT, data_helper=data_helper, excluded_channels=EXCLUDED_CHANNELS)

    if args.wavelet:
        handle_morlet_wavelet_transform(state=args.classes, root_path=IMAGE_ROOT, data_helper=data_helper, excluded_channels=EXCLUDED_CHANNELS)


if __name__ == '__main__':
    main()