from time import time
import mne
from mne.preprocessing import ICA
import os
import matplotlib
import matplotlib.pyplot as plt

CWD = os.path.dirname(os.path.realpath(__file__))
FILE = os.path.join(CWD, '..', 'data/sub-pd3/ses-off/eeg/sub-pd3_ses-off_task-rest_eeg.bdf')
#FILE = os.path.join(CWD, '..', 'data/sub-hc7/ses-hc/eeg/sub-hc7_ses-hc_task-rest_eeg.bdf')
# mne.pick_types(raw.info, eeg=True)

# INFO: There might be some discrepencies based on the version of
# MNE I am using, v0.19



def main():
    """
    data_dir = os.path.join(os.path.dirname(mne.__file__), 'channels', 'data')
    for subfolder in ['layouts', 'montages']:
        print('\nBUILT-IN {} FILES'.format(subfolder[:-1].upper()))
        print('======================')
        print(sorted(os.listdir(os.path.join(data_dir, subfolder))))
    exit(0)
    """
    drop_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status']
    channels = [
        'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
        'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
        'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
    ]
    single_drop_channels = [
        'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
        'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
        'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
    ]
    montage = mne.channels.read_montage('biosemi64', ch_names=channels)
    print(montage.ch_names)

    """
    raw = mne.io.read_raw_fif('temp.bdf')
    raw.load_data()
    raw.crop(tmax=60.0)
    raw.plot(order=raw.ch_names, n_channels=len(raw.ch_names))
    exit(0)
    """

    raw = mne.io.read_raw_bdf(FILE, preload=True, stim_channel='auto', verbose=False)
    raw.drop_channels(drop_channels)

    filt_raw = raw.copy()
    reconst_raw = raw.copy()


    filt_raw.load_data().filter(l_freq=1., h_freq=40.0)
    filt_raw.set_montage(montage, set_dig=True)
    filt_raw.crop(tmax=120.0)

    ica = ICA(n_components=32, method='extended-infomax', random_state=1)
    ica.fit(filt_raw)
    ica.plot_components()
    #ica.plot_overlay(filt_raw, exclude=[0], picks='eeg')
    #ica.plot_overlay(filt_raw, exclude=[2], picks='eeg')

    #ica.exclude = [0, 1]
    ica.apply(reconst_raw, exclude=[i for i in range(32)])
    reconst_raw.plot()

    data, times = reconst_raw[:]
    plt.plot(times, data.T)
    plt.show()
    del reconst_raw

if __name__ == '__main__':
    main()