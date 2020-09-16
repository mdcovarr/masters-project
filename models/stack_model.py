"""
    Module to create a stacked model from different channel models
"""
from tensorflow.keras.models import load_model
import os
import glob
import numpy as np
from PIL import Image

CWD = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(CWD, '..', 'test-stft-ensemble-200-40Hz')
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
        filename = os.path.join(CWD, '..', 'archive', 'models', 'stft-200-softmax-40Hz', model_name)
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

    return X, y

def stacked_dataset(members, X):
    """
    Function used to get prediction of data for each model
    and stack results as input for the stacked model
    :param members: list of models, one for each channel of EEG reading
    :param X: list of concatenated images for each channel. 1 data point is a concatenated image
    :return stack_x: output predictions from each of the indvidual channel model, used as input for stacked model
    """
    # there are 32 models
    # images are (200x200x3)
    stack_x = []

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
        stack_x.append(y_hat)

    stack_x = np.array(stack_x)
    return stack_x

def reverse_one_hot(data):
    labels = []

    for i in range(data.shape[0]):
        point = data[i, :]

        if point[0] > 0:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(labels)

def stacked_prediction(stack_x, y):
    """
    Function uses the output of the single models to determine it's final guess
    using majority rule in final output
    """
    y_hat = []

    # stack_x = (32xDATA_POINTSx2)
    for i in range(stack_x.shape[1]):
        # need to determine what is the majority rule for each image
        data = stack_x[:, i, :]

        curr_labels = reverse_one_hot(data)

        if np.sum(curr_labels) > 16.0:
            y_hat.append(1)
        else:
            y_hat.append(0)

    return np.array(y_hat)

def get_accuracy(y_hat, y):
    """
    Function used to get the accuracy of the ensemble model
    """
    correct = 0

    if len(y_hat) != len(y):
        return 0

    for i in range(len(y_hat)):
        if y_hat[i] == y[i]:
            correct += 1

    acc = (1.0 * correct) / len(y_hat)
    return acc

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

    print('----------------- Getting individual model outputs... -------------------')
    stack_x = stacked_dataset(members, X)

    y_hat = stacked_prediction(stack_x, y)

    acc = get_accuracy(y_hat, y)

    print('current accuracy: {0} %'.format(acc * 100))

if __name__ == '__main__':
    main()
