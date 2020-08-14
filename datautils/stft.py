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

class STFT(object):
    """
    Class used to help create spectrogram images via STFT for EEG data
    """
    def __init(self, **kwargs):
        """
        Default constructor
        """
        self.classes = kwargs['state']
        self.root_path = kwargs['root_path']
        self.all_files = kwargs['all_files']

    def iterate_eeg_data(self, **kwargs):
        """
        Function used to iterate data, and generate the spectrogram images
        from the short time fourier transform of an EEG signal
        """
        pass

    def generate_stft_transform(self):
        """
        Function used to generate the short-time fourier transform
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

    def handle_PD_patients(self, **kwargs):
        """
        Funtion used to handle PD patients specifically.
        """
        # Need to iterate through all patient data
        class_root = os.path.join(kwargs['root_path'], 'PD')
        clean_and_create(class_root)

        for subject_name in kwargs['all_files']['PD']:
            subject_files = kwargs['all_files']['PD'][subject_name]

            for filename in subject_files:
                """
                    1. Need to load in the data
                """
                data = load_data(filename)

                # apply filters
                data.filter(0.5, 32.0, fir_design='firwin')

                """
                    2. Create output dir for patient data
                """
                patient_path = get_patient_path(filename, class_root, 'PD')
                clean_and_create(patient_path)

                """
                    3. Create Spectrogram images from the data
                """
                stft_iterate_eeg_data(data=data, output_dir=patient_path)

    def handle_NONPD_patients(self, **kwargs):
        """
        Function used to handle NONPD patients specifically.
        """
        # Make directory for class (e.g., NONPD)
        class_root = os.path.join(kwargs["root_path"], 'NONPD')
        clean_and_create(class_root)

        # need to read every patient EEG reading
        for filename in kwargs["all_files"]['NONPD']:
            """
                1. Need to load in the data
            """
            data = load_data(filename)

            # apply filters
            data.filter(0.5, 32.0, fir_design='firwin')

            """
                2. Create output dir for patient data
            """
            patient_path = get_patient_path(filename, class_root, 'NONPD')
            clean_and_create(patient_path)

            """
                3. Create spectrogram images from the data
            """
            stft_iterate_eeg_data(data=data, output_dir=patient_path)

