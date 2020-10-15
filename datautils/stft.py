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


class STFT(object):
    """
    Class used to help create spectrogram images via STFT for EEG data
    """
    def __init__(self, **kwargs):
        """
        Default constructor
        """
        self.classes = kwargs['state']
        self.root_path = kwargs['root_path']
        self.data_helper = kwargs['data_helper']
        self.excluded_channels = kwargs['excluded_channels']

        # TODO: can make the following parameters passed as options
        #       for wavelet transform
        self.segment_time_size = 2
        self.band_filter = [0.5, 40.0]
        self.vmin_vmax = [0.0, 5.0]
        self.nperseg = 256
        self.noverlap = 128

        # Parameter to determine if data has been preprocessed via ICA
        self.ica_preprocessed = True

    def generate_stft_transform(self):
        """
        Function used to generate the short-time fourier transform
        """
        self.data_helper.clean_create_dir(self.root_path)

        for class_label in self.data_helper.all_files.keys():
            class_files = self.data_helper.all_files[class_label]

            class_root = os.path.join(self.root_path, str(class_label))
            self.data_helper.clean_create_dir(class_root)

            for eeg_file in class_files:
                # Load EEG data

                if self.ica_preprocessed:
                    raw = self.data_helper.load_fif_data(eeg_file)
                else:
                    raw = self.data_helper.load_data(eeg_file)
                    # Apply filter to data
                    raw.filter(self.band_filter[0], self.band_filter[1], fir_design='firwin')

                # Create output dir to patient data
                filename_dir = eeg_file.split(os.path.sep)[-4:-3]
                patient_path = os.path.join(class_root, '/'.join(filename_dir))
                self.data_helper.clean_create_dir(patient_path)

                # iterate EEG data for patient
                self.iterate_eeg_data(data=raw, output_dir=patient_path)

    def iterate_eeg_data(self, **kwargs):
        """
        Function used to iterate data, and generate the spectrogram images
        from the short time fourier transform of an EEG signal
        """
        raw_data = kwargs["data"].copy()
        kwargs["data"].load_data()
        data = kwargs["data"].to_data_frame()

        # Get list of channel names
        channel_names = raw_data.ch_names

        # Sample Frequency
        fs = int(raw_data.info["sfreq"])

        # STFT Parameters
        segment_size = fs * self.segment_time_size

        for channel in channel_names:
            if channel in self.excluded_channels:
                continue

            # Create channel output directory and iterate through all channels
            channel_path = os.path.join(kwargs["output_dir"], channel)
            self.data_helper.clean_create_dir(channel_path)

            # counter for image names e.g. ~/masters-project/spectrogram-images/NONPD/sub-hc4/ses-hc/Fp1/0'
            image_counter = 0

            channel_data = data[channel].values
            size = len(channel_data)
            segments = int(size // segment_size)

            for index in range(segments):
                lower_point = index * segment_size
                upper_point = lower_point + segment_size - 1
                current_segment = channel_data[lower_point : upper_point]

                f, t, Zxx = signal.stft(current_segment, fs, window='blackman', nperseg=self.nperseg, boundary=None, noverlap=self.noverlap)

                Zxx = Zxx[0 : 25]
                f = f[0 : 25]

                try:
                    output_filepath = os.path.join(channel_path, str(image_counter))
                    plt.pcolormesh(t, f, np.abs(Zxx), vmin=self.vmin_vmax[0], vmax=self.vmin_vmax[1], shading='gouraud')

                    plt.axis('off')

                    figure = plt.gcf()
                    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.clf()

                    image_counter += 1
                except FloatingPointError as e:
                    print('Caught divide by 0 error: {0}'.format(output_filepath))
