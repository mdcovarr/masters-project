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
        self.all_files = kwargs['all_files']

    def generate_wavelet_transform(self):
        """
        Function used to genereate the wavelet transform images from data
        """

        if (self.classes == 'ALL'):
            class_list = ['NONPD', 'PD']
        else:
            class_list = [self.classes]

        if os.path.isdir(self.root_path):
            shutil.rmtree(self.root_path, ignore_errors=True)

        os.makedirs(self.root_path)

        for curr_class in class_list:
            if curr_class == 'PD':
                self.handle_PD_patients()

            if curr_class == 'NONPD':
                self.handle_NONPD_patients()

    def handle_NONPD_patients(self):
        """
        Function used to handle the non PD patient files
        """
        class_root = os.path.join(self.root_path, 'NONPD')

        if os.path.isdir(class_root):
            shutil.rmtree(class_root, ignore_errors=True)

        os.makedirs(class_root)

        for filename in self.all_files['NONPD']:
            """
                1. Need to lead in the data
            """
            raw = mne.io.read_raw_bdf(filename, preload=True, stim_channel='auto', verbose=False)
            raw.filter(0.5, 32.0, fir_design='firwin')

            """
                2. Create output dir for patient data
            """
            filename_dir = filename.split(os.path.sep)[-4:-2]
            patient_path = os.path.join(class_root, '/'.join(filename_dir))

            if os.path.isdir(patient_path):
                shutil.rmtree(patient_path, ignore_errors=True)

            os.makedirs(patient_path)

            """
                3. Create transform images
            """
            self.iterate_eeg_data(data=raw, output_dir=patient_path)

    def handle_PD_patients(self):
        """
        Function used to handle PD patient files
        """
        # Need to iterate through all patient data
        class_root = os.path.join(self.root_path, 'PD')

        if os.path.isdir(class_root):
            shutil.rmtree(class_root, ignore_errors=True)

        os.makedirs(class_root)

        for subject_name in self.all_files['PD']:
            subject_files = self.all_files['PD'][subject_name]

            for filename in subject_files:
                """
                    1. Need to load in the data
                """
                raw = mne.io.read_raw_bdf(filename, preload=True, stim_channel='auto', verbose=False)
                raw.filter(0.5, 32.0, fir_design='firwin')

                """
                    2. Create output dir for patient data
                """
                filename_dir = filename.split(os.path.sep)[-4:-2]
                patient_path = os.path.join(class_root, '/'.join(filename_dir))

                if os.path.isdir(patient_path):
                    shutil.rmtree(patient_path, ignore_errors=True)

                os.makedirs(patient_path)


                """
                    3. Create Spectrogram images from the data
                """
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

    def eeg_to_coefficients(self, **kwargs):
        """
            Function used to perform wavelet transform and save the coefficient information
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
            if channel == 'Status':
                continue

            channel_path = os.path.join(kwargs['output_dir'], channel)
            df = pandas.DataFrame()

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
                coefficients, frequencies = pywt.cwt(np.array(current_segment), scales, 'cmor0.4-1.0')

                vmin = abs(coefficients).min()
                vmax = abs(coefficients).max()

                coefficients = abs(coefficients)

                coefficients = coefficients.reshape((-1, coefficients.shape[0] * coefficients.shape[1]))

                df = df.append(pandas.DataFrame(data=coefficients))

            # Now we need to output to file with data
            output_file = os.path.join(channel_path, 'data.csv')
            df.to_csv(output_file)