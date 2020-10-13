#!/usr/bin/env python3
"""
    Script used to read EEG data and perform ICA
"""
import os
import mne
from mne.preprocessing import ICA
import matplotlib
import matplotlib.pyplot as plt

class DataICA(object):
    """
        Class used to perform ICA on input EEG data
    """
    def __init__(self, **kwargs):
        self.data_helper = kwargs['data_helper']
        self.drop_channels = kwargs['exclude_channels']
        self.root_path = kwargs['root_path']

    def generate_ica(self):
        self.data_helper.clean_create_dir(self.root_path)

        for class_label in self.data_helper.all_files.keys():
            class_files = self.data_helper.all_files[class_label]
            class_root = os.path.join(self.root_path, str(class_label))
            self.data_helper.clean_create_dir(class_root)

            for eeg_file in class_files:
                # Load EEG data
                # need to creat the correct output directory
                raw = self.data_helper.load_data(eeg_file)

                # drop channels
                raw.drop_channels(self.drop_channels)

                # apply filters
                raw.filter(l_freq=1.0, h_freq=40.0)

                # get Montage
                montage = mne.channels.read_montage('biosemi64', ch_names=raw.ch_names)
                raw.set_montage(montage, set_dig=True)

                # ICA analysis
                ica = ICA(n_components=20, method='extended-infomax', random_state=92)
                ica.fit(raw)
                ica.apply(raw, exclude=[0, 1, 2])

                # Need to save new data file
                file_path = eeg_file.split(os.path.sep)[-4:-3]
                filename = os.path.basename(eeg_file)
                patient_path = os.path.join(class_root, '/'.join(file_path))
                self.data_helper.clean_create_dir(patient_path)

                full_file_path = os.path.join(patient_path, filename)
                raw.save(full_file_path, overwrite=True)
                del raw
