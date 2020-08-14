import mne
import glob
import argparse
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pandas
import os
import re
from scipy import signal
from scipy.fftpack import fft, ifft
import pywt
from PyEMD import EMD

EXCLUDE_CHANNELS = ['Status', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']

class WaveletTransform(object):
    """
    Class used to help create wavelet transform of EEG data
    """
    def __init__(self, **kwargs):
        """
        Default constructor
        """
        self.classes = kwargs['state']
        self.root_path = kwargs['root_path']
        self.data_helper = kwargs['data_helper']

    def generate_wavelet_transform(self):
        """
        Function used to genereate the wavelet transform images from data
        """
        self.data_helper.clean_create_dir(self.root_path)

        for class_label in self.data_helper.all_files.keys():
            class_files = self.data_helper.all_files[class_label]

            class_root = os.path.join(self.root_path, str(class_label))
            self.data_helper.clean_create_dir(class_root)

            for eeg_file in class_files:
                # Load EEG data
                raw = self.data_helper.load_data(eeg_file)

                # Apply filter to data
                raw.filter(0.5, 32.0, fir_design='firwin')

                # Create output dir to patient data
                filename_dir = eeg_filename.split(os.path.sep)[-4:-2]
                patient_path = os.path.join(class_root, '/'.join(filename_dir))
                self.data_helper.clean_create_dir(patient_path)

                # iterate EEG data for patient
                self.iterate_eeg_data(data=raw, output_dir=patient_path)

    def iterate_eeg_data(self, **kwargs):
        """
        Function used to iterate data, and generate the wavelet transform images
        """
        raw_data = kwargs['data'].copy()
        kwargs['data'].load_data()
        data = kwargs['data'].to_data_frame()

        # Get list of channel  names
        channel_names = raw_data.ch_names

        # Sample Frequency
        fs = int(raw_data.info['sfreq'])

        # Status channel
        status_data = data['Status'].values

        # Wavelet Transform Parameters
        segment_size = 1024 # 2 seconds

        for channel in channel_names:
            if channel in EXCLUDE_CHANNELS:
                continue

            channel_path = os.path.join(kwargs['output_dir'], channel)

            if os.path.isdir(channel_path):
                shutil.rmtree(channel_path, ignore_errors=True)

            os.makedirs(channel_path)

            # counter for image names
            image_counter = 0

            channel_data = data[channel].values
            size = len(channel_data)
            segments = int(size // segment_size)
            image_counter = 0

            for index in range(segments):
                lower_point = index * segment_size
                upper_point = lower_point + segment_size
                current_segment = channel_data[lower_point : upper_point]

                scales = np.arange(1, 32)

                # cmor0.4-1.0
                coef, freq = pywt.cwt(np.array(current_segment), scales, 'cmor0.4-1.0')

                vmin = 0.0
                vmax = 30.0

                coef = np.flip(coef, axis=0)

                try:
                    output_file = os.path.join(channel_path, str(image_counter))

                    plt.pcolormesh(abs(coef), vmax=vmax, vmin=vmin)
                    plt.show()

                    """
                        Modifying Plot settings
                    """
                    plt.axis('off')
                    figure = plt.gcf()
                    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.clf()

                    image_counter += 1
                except FloatingPointError as e:
                    print('Caught divide by 0 error: {0}'.format(output_filepath))
