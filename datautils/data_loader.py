"""
    Script to help load EEG data files, separating files by class type
"""

import mne
import glob
import pandas
import shutil
import numpy as np
import math
import os
import re
from scipy import signal

CWD = os.path.dirname(os.path.realpath(__file__))

class DataLoader(object):
    """
        Data Loader class to help import data and separate by class
    """
    def __init__(self, **kwargs):
        """
        Default Constructor
        """
        self.paths = kwargs['paths']

    def load_data_files(self, locations):
        """
        Function used to get all files of interest in regards to
        EEG data. Locations is a list of different paths where each entry
        is a path for a certain class
        :param locations: list of location paths for each class
        """
        all_files = {}

        for index, location in enumerate(locations):
            class_files = glob.glob(location, recursive=True)
            all_files[str(index)] = class_files

        self.all_files = all_files

    def clean_create_dir(self, path):
        """
        Function used to clear out a directory nd create a new one
        :param path: directory path to clean and create
        """
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)

        os.makedirs(path)

    def load_data(self, filename):
        """
        Function used to load EEG data
        :param filename: the file name of .bdf file we want to read in
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