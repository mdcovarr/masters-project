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

CWD = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(CWD, 'data')
SPECTROGRAM_ROOT = os.path.join(CWD, 'spectrogram-images-32Hz')
WAVELET_ROOT = os.path.join(CWD, 'wavelet-images')
PATTERN = os.path.join(ROOT_PATH, '**', '*.bdf')
TEST_FILE = './data/sub-hc1/ses-hc/eeg/sub-hc1_ses-hc_task-rest_eeg.bdf'
NONPD_PATH = r'.*sub-hc\d{1,2}.*'
PD_PATH = r'.*sub-pd\d{1,2}.*'

EXCLUDED_CHANNELS = ['Status', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
MY_DPI = 192

INTERVAL = 2560
FREQUENCY = 512
M = 256
MAX_AMP = 2
CHANNEL_COUNT = 40

"""
    Wavelet Transform Parameters
"""
W = 6.
FREQ = np.linspace(1, FREQUENCY/2, 100)
WIDTHS = W * FREQUENCY / (2 * FREQ * np.pi)

# TODO: Parameters that can be added
# Segemnt duration
# high pass filter
# low pass filter
# overlap
# number of data pts per segment

def handle_arguments():
    """
    Function used to set arguments that can be passed to the script
    :return args: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Split EEG data preprocess and create spectrograms')

    parser.add_argument('-c', '--class', dest='classes', choices=['PD', 'NONPD', 'ALL'],
            help='Flag used to determine what class type we want to cretae spectrogram images for')
    parser.add_argument('-t', '--test', dest='test', action="store_true", default=False,
            help='Flag used to test software. Thus, only a single file will be analyzed')
    parser.add_argument('-s', '--stft', dest='stft', action='store_true', default=False,
            help='Flag used to utilize the short-time fourier transform in data processing')
    parser.add_argument('-w', '--wave', dest='wavelet', action='store_true', default=False,
            help='Flag used to utilize wavelet transform in data processing')

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
    :return raw: the raw data of given filename
    """
    raw = mne.io.read_raw_bdf(filename, preload=True, stim_channel='auto', verbose=False)

    """
        Informative parameters

        raw_data.info["sfreq"]
        raw_data.info["nchan"]
        raw_data.ch_names
    """

    return raw

def get_patient_path(filename, class_root, label):
    """
    Function used to get the patients location to place spectrogram images
    """
    filename_dir = filename.split(os.path.sep)[-4:-2]
    patient_path = os.path.join(class_root, '/'.join(filename_dir))

    return patient_path

def stft_iterate_eeg_data(**kwargs):
    """
    Function to iterate data and create stft spectrogram images
    """
    # Data to generate STFT for.
    raw_data = kwargs["data"].copy()
    kwargs["data"].load_data()
    data = kwargs["data"].to_data_frame()

    # Get list of channel names
    channel_names = raw_data.ch_names

    # Sample Frequency
    fs = int(raw_data.info["sfreq"])

    # STFT Parameters
    segment_size = 1024 # 4 seconds
    amp = 1 * np.sqrt(2)


    for channel in channel_names:
        if channel in EXCLUDED_CHANNELS:
            continue
        # Create channel output directory and iterate through all channels
        channel_path = os.path.join(kwargs["output_dir"], channel)
        clean_and_create(channel_path)

        # counter for image names e.g. ~/masters-project/spectrogram-images/NONPD/sub-hc4/ses-hc/Fp1/0'
        image_counter = 0

        channel_data = data[channel].values
        size = len(channel_data)
        segments = int(size // segment_size)

        for index in range(segments):
            lower_point = index * segment_size
            upper_point = lower_point + segment_size
            current_segment = channel_data[lower_point : upper_point]

            f, t, Zxx = signal.stft(current_segment, fs, window='blackman', nperseg=256, boundary=None, noverlap=230)

            Zxx = Zxx[0 : 16]
            f = f[0 : 16]
            print(max(Zxx))

            try:
                output_filepath = os.path.join(channel_path, str(image_counter))
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
                plt.axis('off')

                figure = plt.gcf()
                figure.set_size_inches(1.69, 1.69)
                plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, dpi=100)
                plt.clf()

                image_counter += 1
            except FloatingPointError as e:
                print('Caught divide by 0 error: {0}'.format(output_filepath))

def handle_morlet_wavelet_transform(**kwargs):
    """
    Function used to handle the creating of spectrogram images
    using wavelet transform
    """
    wavelet_helper = wavelet_transform.WaveletTransform(**kwargs)
    wavelet_helper.generate_wavelet_transform()

def handle_PD_patients(**kwargs):
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

def handle_NONPD_patients(**kwargs):
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


def handle_stft(**kwargs):
    """
    Function used to handle the creation of spectrogram images
    """
    class_list = []

    if (kwargs["state"] == 'ALL'):
        class_list = ['NONPD', 'PD']
    else:
        class_list = [kwargs["state"]]

    # need to check if output directories exist, create new
    clean_and_create(kwargs["root_path"])

    for curr_class in class_list:

        if curr_class == 'PD':
            handle_PD_patients(**kwargs)

        if curr_class == 'NONPD':
            handle_NONPD_patients(**kwargs)

def handle_test(**kwargs):
    """
    Function to run test graphs
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
            raw = load_data(filename)

            """
                Informative parameters

                raw_data.info["sfreq"]
                raw_data.info["nchan"]
                raw_data.ch_names
            """
            sfreq = raw.info['sfreq']
            sfreq = raw.info['sfreq']
            data, times = raw[:5, int(sfreq * 1):int(sfreq * 10)]

            fig = plt.subplots(figsize=(10,8))
            plt.plot(times, data.T);
            plt.xlabel('Seconds')
            plt.ylabel('mu V')
            plt.title('Channels: 1-5');
            plt.legend(raw.ch_names[:5]);
            plt.show()

            raw.filter(1.0, 60.0, fir_design='firwin', skip_by_annotation='edge')
            data_f, times_f = raw[:5, int(sfreq * 1):int(sfreq * 10)]

            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15, 10))

            ax1.plot(times, data.T);
            ax1.set_title('Before Filter')
            ax1.set_xlabel('Seconds')
            ax1.set_ylabel('mu V')

            ax2.plot(times_f, data_f.T);
            ax2.set_title('After Filter')
            ax2.set_xlabel('Seconds')

            plt.legend(raw.ch_names[:5], loc=1);
            plt.show()

            raw.plot(duration=60, block=True)

            raw.plot_psd(tmax=np.inf, fmax=250)

            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                        exclude='bads')

            # events.shap|e
            events = mne.find_events(raw, shortest_event=0, stim_channel='Status', verbose=False)

            epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True, verbose=False)

            # remove channels that aren't eeg electrodes we care about
            epochs.drop_channels(ch_names=['Nose', 'REOG', 'LEOG', 'IEOG', 'SEOG', 'M1', 'M2','EXG8'])

            # Export data in tabular structure as a pandas DataFrame.
            epochs_df = epochs.to_data_frame()


            evoked = epochs['target'].average()
            evoked.plot();
            exit(0)


def separate_data_per_patient(pd_list, pattern):
    """
    Function used to group together the data of each PD patient per patient
    """
    pd_list_indexs = [3, 5, 6, 9, 11, 12, 13, 14, 16, 17, 19, 22, 23, 26, 28]

    pd_patient_list = {}

    for index in pd_list_indexs:
        key = '{0}{1}'.format(pattern, index)

        pd_patient_list[key] = [i for i in pd_list if key in i]

    return pd_patient_list

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

    """
        4. Can separate each PD patient to it's respective ON Medication and OFF medication recordings
    """
    pd_patient_list = separate_data_per_patient(pd_list, 'sub-pd')

    all_files["PD"] = pd_patient_list

    # Function used for the short time fourier transform
    if args.stft:
        handle_stft(state=args.classes, root_path=SPECTROGRAM_ROOT, all_files=all_files)

    if args.wavelet:
        handle_morlet_wavelet_transform(state=args.classes, root_path=WAVELET_ROOT, all_files=all_files)

    if args.test:
        handle_test(state=args.classes, root_path=SPECTROGRAM_ROOT, all_files=all_files)


if __name__ == '__main__':
    main()