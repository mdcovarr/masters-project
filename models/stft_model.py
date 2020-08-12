import matplotlib
matplotlib.use('Agg')

# sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import glob
import random
import os


ACTIVATION = 'relu'
PREDICT_ACTIVATION = 'sigmoid'
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']

CWD = os.path.dirname(os.path.realpath(__file__))
PD_ROOT = os.path.join(CWD, '..', 'spectrogram-images', 'PD', '**')
PD_OFF_ROOT = os.path.join(CWD, '..', 'spectrogram-images', 'PD', '**', 'ses-off')
PD_ON_ROOT = os.path.join(CWD, '..', 'spectrogram-images', 'PD', '**', 'ses-on')
NONPD_ROOT = os.path.join(CWD, '..', 'spectrogram-images', 'NONPD', '**')
PATH_TO_DATASET = [PD_OFF_ROOT, NONPD_ROOT]
CLASSES = 2

CHANNEL_CHOICES = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
    'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
    'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

def handle_arguments():
    """
    Function used to parse script arguments
    :return args: commandline arguments for script
    """

    parser = argparse.ArgumentParser(description='Train a model to classify spectrograms')
    parser.add_argument('-c', '--channel', dest='channel', required=True, choices=CHANNEL_CHOICES,
                        help='Flag used to determine what channel we want to create a model for')
    parser.add_argument('-s', '--set', dest='size', required=True,
                        help='Flag used to determine the amount of experiments to import for train/test data')
    parser.add_argument('-e', '--epochs', dest='epochs', required=True,
                        help='Flag used to determine the number of epochs for training')
    args = parser.parse_args()

    return args


def determine_data_paths(paths_to_datasets, channel, size):
    """
    Function used to determine full paths to datasets we will be reading for training and testing
    :param paths_to_datasets: path to root of data for PD and NONPD
    :param channel: channel we want to train a model for
    :param size: number of experiments we want to input for test/train data
    :return all_paths: all paths to datasets
    """
    all_paths = {}
    class_count = 0

    for path_to_dataset in paths_to_datasets:
        path_with_channel = os.path.join(path_to_dataset, str(channel))
        dirs = glob.glob(path_with_channel, recursive=True)

        image_paths = sorted(list(dirs))
        image_paths = image_paths[:size]

        all_paths[str(class_count)] = image_paths
        class_count += 1

    return all_paths


def get_train_test_data(all_data_paths):
    """
    Function used to get all train and test images and label data
    :param all_data_paths: dictionary of paths for data
    :return data, labels: data with their labels (e.g., NONPD, PD on medication, PD off medication)
    """
    data = []
    labels = []

    for key in all_data_paths.keys():
        data_paths = all_data_paths[key]

        # Need to iterate through all paths of current class and import data
        for data_path in data_paths:
            image_files = glob.glob(os.path.join(data_path, '*'))

            for image_file in image_files:
                image_src = np.array(Image.open(image_file).convert('RGB'))

                data.append(image_src)
                labels.append(int(key))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


def main():
    """
    Main Enterance of model
    """

    # handle arguments
    args = handle_arguments()

    print('-------------------------\n[INFO] Preprocessing Data\n-------------------------')

    # Get all paths we want to read data from for classes: NONPD, PD medication, PD no medication
    all_data_paths = determine_data_paths(PATH_TO_DATASET, args.channel, int(args.size))

    data_set, labels = get_train_test_data(all_data_paths)

    # Normalize dataset
    data_set = data_set / 255.0
    # One-Hot encode labels
    labels = to_categorical(labels, CLASSES)

    print('-------------------------\n[INFO] Building Model\n-------------------------')

    # split training and test data
    (trainX, testX, trainY, testY) = train_test_split(data_set, labels, test_size=0.25, random_state=42)

    # building model
    model = Sequential()

    # adding layers
    model.add(Conv2D(16, 3, input_shape=(130, 130, 3), activation=ACTIVATION))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, activation=ACTIVATION))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, activation=ACTIVATION))
    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dense(500, activation=ACTIVATION))

    model.add(Dense(2, activation=PREDICT_ACTIVATION))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    print('-------------------------\n[INFO] Train Model\n-------------------------')

    history = model.fit(trainX, trainY, epochs=int(args.epochs), validation_data=(testX, testY))

    print('-------------------------\n[INFO] Plot Results\n-------------------------')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

    print('Test Loss: {test_loss}'.format(test_loss=test_loss))
    print('Test Accuracy: {test_acc}'.format(test_acc=test_acc))
    print(test_acc)


if __name__ == '__main__':
    main()