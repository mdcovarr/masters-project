import matplotlib
matplotlib.use('Agg')

# sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
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
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

CWD = os.path.dirname(os.path.realpath(__file__))
PD_ROOT = os.path.join(CWD, '..', 'wavelet-images', 'PD', '**')
NONPD_ROOT = os.path.join(CWD, '..', 'wavelet-images', 'NONPD', '**')
PATH_TO_DATASET = [PD_ROOT, NONPD_ROOT]


def handle_arguments():
    """
    Function used to parse script arguments
    :return args: commandline arguments for script
    """
    """
    parser.add_argument("-d", "--dataset", required=True,
            help="path to input dataset of images")
    parser.add_argument("-m", "--model", required=True,
            help="path to output trained model")
    parser.add_argument("-l", "--label-bin", required=True,
            help="path to output label binarizer")
    parser.add_argument("-p", "--plot", required=True,
            help="path to output accuracy/loss plot")
    """
    channel_choices = [
        'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3',
        'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8',
        'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'EXG1', 'EXG2', 'EXG3',
        'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'
    ]

    parser = argparse.ArgumentParser(description='Train a model to classify spectrograms')
    parser.add_argument('-c', '--channel', dest='channel', required=True, choices=channel_choices,
                        help='Flag used to determine what channel we want to create a model for')
    parser.add_argument('-s', '--set', dest='size', required=True,
                        help='Flag used to determine the amount of experiments to import for train/test data')
    parser.add_argument('-i', '--image-size', dest='image_size', required=True,
                        help='Flag used to determine the length and width to resize the data spectrogram images')
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
    all_paths = []

    for path_to_dataset in paths_to_datasets:
        path_with_channel = os.path.join(path_to_dataset, str(channel))
        dirs = glob.glob(path_with_channel, recursive=True)

        image_paths = sorted(list(dirs))
        image_paths = image_paths[:size]

        all_paths.extend(image_paths)

    return all_paths


def get_all_csv_paths(all_root_paths):
    """
    Function used to get all csv data files
    example: './NONPD/sub-hc1/ses-hc/Fp1/' gets all csv's at path NONPD/sub-hc1/ses-hc/Fp1/*.csv'
    :param all_root_paths:
    :return all_image_paths: complete paths to all images we are going to input for train/test data
    """
    all_csv_paths = []

    for root_path in all_root_paths:
        csv_path = os.path.join(root_path, '*.csv')
        csv_paths = glob.glob(csv_path, recursive=True)

        all_csv_paths.extend(csv_paths)

    return all_csv_paths

def get_train_test_data(csv_path_list, image_size):
    """
    Function used to get all train and test images and label data
    :param images_path_list: list of all images we want to read
    :param image_size: image resize to save computation time
    :return data, labels: data with their labels (e.g., DROWSY, FOCUSED, UNFOCUSED)
    """
    data = np.empty((0, 0))
    labels = np.empty((0, 0))

    for csv_path in csv_path_list:
        # image_path: NONPD/.../data.csv
        # TODO: Need to read content from csv files
        curr_data = np.genfromtxt(csv_path, delimiter=',')

        # all rows for a given class for a given channel
        curr_data = np.delete(curr_data, 0, 0)
        data_len = len(curr_data)

        if data.shape[0] == 0:
            data = curr_data
        else:
            data = np.vstack((data, curr_data))

        # determine the label
        if 'PD' in csv_path:
            label = 1
        else:
            label = 0

        label_list = [label for _ in range(data_len)]

        if labels.shape[0] == 0:
            labels = np.array(label_list)
        else:
            labels = np.append(labels, np.array(label_list))

    return data, labels


def main():
    """
    Main Enterance of model
    """

    # handle arguments
    args = handle_arguments()

    print('-------------------------\n[INFO] Preprocessing Data\n-------------------------')

    # Get all paths we want to read data from
    data_paths = determine_data_paths(PATH_TO_DATASET, args.channel, int(args.size))

    all_image_paths = get_all_csv_paths(data_paths)

    data_set, labels = get_train_test_data(all_image_paths, int(args.image_size))

    print('-------------------------\n[INFO] Building Model\n-------------------------')

    # split training and test data
    (trainX, testX, trainY, testY) = train_test_split(data_set, labels, test_size=0.25, random_state=42)

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # building model
    model = Sequential()

    # adding layers
    model.add(Conv2D(32, (3, 3), input_shape=(int(args.image_size), int(args.image_size), 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation=ACTIVATION))

    # prediction layer, using softmax because we are expecting more than two outcomes (DROWSY, FOCUSED, UNFOCUSED)
    model.add(Dense(3, activation=PREDICT_ACTIVATION))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    print('-------------------------\n[INFO] Train Model\n-------------------------')

    history = model.fit(trainX, trainY, epochs=int(args.epochs), validation_data=(testX, testY), batch_size=300)

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