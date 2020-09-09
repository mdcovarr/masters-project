"""
    Module to create a stacked model from different channel models
"""
from tensorflow.keras.models import load_model
import os
import glob
import numpy as np
from PIL import Image

CWD = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(CWD, '..', 'stft-ensemble-200-40Hz')
SUFFIX_CHANNELS = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
    'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

def load_all_models(model_channels):
    """
    Function used to load all the models created. One for each channel
    of the EEG headset
    :param model_channels: list of channels models were created for
    """
    all_models = []
    model_channels.sort()

    for channel in model_channels:
        model_name = 'model.{0}'.format(channel)
        filename = os.path.join(CWD, '..', 'archive', 'models', 'stft_40Hz', model_name)
        model = load_model(filename)
        all_models.append(model)

    # models should be listen in same order as images concatenated
    return all_models

def load_data(data_dir):
    """
    Function to load the data of concatenated images
    """
    # currently will load only a portion of the data
    data_dir = os.path.join(data_dir, '*')
    dirs = glob.glob(data_dir)
    nonpd_dirs = []
    pd_dirs = []

    # create list of NONPD directories
    for i in range(len(dirs)):
        if 'sub-hc' in dirs[i]:
            nonpd_dirs.append(dirs[i])
        else:
            pd_dirs.append(dirs[i])

        i += 1

    # sort and shorten both lists
    nonpd_dirs.sort()
    pd_dirs.sort()

    nonpd_dirs = nonpd_dirs[:1]
    pd_dirs = pd_dirs[:1]

    data_dirs = []
    data_dirs.extend(nonpd_dirs)
    data_dirs.extend(pd_dirs)

    # load data from the subset of dirs
    X = None
    y = None
    for data_dir in data_dirs:
        image_data = glob.glob(os.path.join(data_dir, '*'))

        for image in image_data:
            image_src = np.asarray(Image.open(image))
            image_src = image_src.reshape((1, image_src.shape[0] * image_src.shape[1], 3))

            if 'sub-hc' in image:
                image_label = np.array([0])
            else:
                image_label = np.array([1])

            if X is None:
                X = image_src
            else:
                try:
                    X = np.vstack((X, image_src))
                except ValueError:
                    print(X.shape)

            if y is None:
                y = image_label
            else:
                y = np.vstack((y, image_label))

    # dimensions are = (180, 524288, 3) where 180 is the number of data points
    return X, y

def stacked_dataset(members, X):
    """
    Function used to get prediction of data for each model
    and stack results as input for the stacked model
    :param members: list of models, one for each channel of EEG reading
    """
    # there are 32 models
    # images are (200x200x3)
    for i, member in enumerate(members):
        all_data = []

        for j in range(X.shape[0]):
            lower_point = i * 40000
            upper_point = lower_point + 40000
            data_point = X[j, lower_point : upper_point, :]
            data_point = data_point.reshape((200, 200, 3))
            all_data.append(data_point)

        all_data = np.array(all_data)
        y_hat = member.predict(all_data)

def main():
    """
    Script start
    """
    print('----------------- Loading all models... -------------------')
    members = load_all_models(SUFFIX_CHANNELS)
    print('----------------- All models loaded -------------------')

    print('----------------- Loading test train data... -------------------')
    X, y = load_data(DATA_ROOT)
    print('----------------- All data loaded... -------------------')

    stacked_dataset(members, X)

if __name__ == '__main__':
    main()
