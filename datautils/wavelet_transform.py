import mne
import glob
import argparse
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import re
import scipy

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

        # Wavelet Transform Parameters
        segment_size = 2048 # 4 seconds

        for channel in channel_names:
            if channel == 'Status':
                continue

            channel_path = os.path.join(kwargs['output_dir'], channel)

            if os.path.isdir(channel_path):
                shutil.rmtree(channel_path, ignore_errors=True)

            os.makedirs(channel_path)

            # counter for image names
            image_counter = 0

            channel_data = data[channel].values
            size = len(channel_data)
            segment = int(size // segment_size)

            for index in range(segments):
                lower_point = index * segment_size
                upper_point = lower_point + segment_size - 1
                current_segment = channel_data[lower_point : upper_point]

                # Need to perform the wavelet transform
