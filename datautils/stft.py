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
