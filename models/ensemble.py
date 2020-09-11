import matplotlib
matplotlib.use('Agg')

# sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv3D, MaxPool3D, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD

from subprocess import check_call
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil
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
DATA_ROOT = ''
PD_OFF_ROOT = ''
PD_ON_ROOT = ''
NONPD_ROOT = ''
PATH_TO_DATASET = []
CLASSES = 2
README = 'README.md'
ACCURACY_FILE = 'accuracy.png'
LOSS_FILE = 'loss.png'

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
    parser.add_argument('-i', '--image-size', dest='image_size', required=True)
    parser.add_argument('-s', '--set', dest='size', required=True,
                        help='Flag used to determine the amount of experiments to import for train/test data')
    parser.add_argument('-e', '--epochs', dest='epochs', required=True,
                        help='Flag used to determine the number of epochs for training')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
                        help='directory where to output all models created for each channel')
    parser.add_argument('-d', '--data-root', dest='data_root', required=True,
                        help='root of directory for training testing data images')
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


def get_train_test_data(all_data_paths, image_size):
    """
    Function used to get all train and test images and label data
    :param all_data_paths: dictionary of paths for data
    :param image_size: size to reshape input image to
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
                image_src = np.array(Image.open(image_file).resize((image_size, image_size)).convert('RGB'))

                data.append(image_src)
                labels.append(int(key))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def print_model_metadata(readme_file, model, args):
    """
    Function used to print metadata about models to README.md file
    :param readme_file: readme file to print model information
    :param model: current model being trained
    :param args: command line arguments
    """
    output_str = '# Models Metadata\n```\n'

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    output_str += short_model_summary
    output_str += '\n'

    output_str +=  'Epochs: {0}\nImage Size: {1}\n```\n'.format(args.epochs, args.image_size)

    command = 'echo \'{0}\' >> {1}'.format(output_str, readme_file)

    try:
        check_call(command, shell=True)
    except:
        print('Error writing metadata to README file\nexiting...')
        exit(1)

def main():
    """
    Main Enterance of model
    """

    # handle arguments
    args = handle_arguments()

    """
        First determine the root directory of training/testing data for models
    """
    DATA_ROOT = args.data_root
    NONPD_ROOT = os.path.join(CWD, '..', DATA_ROOT, '0', '**')
    PD_OFF_ROOT = os.path.join(CWD, '..', DATA_ROOT, '1', '**')
    PD_ON_ROOT = os.path.join(CWD, '..', DATA_ROOT, '2', '**')
    PATH_TO_DATASET = [NONPD_ROOT, PD_OFF_ROOT]
    print_metadata = True

    print('-------------------------\n[INFO] Preprocessing Data\n-------------------------')

    """
        Create Directory for all models. Creation of a model for each channel
    """
    args.output_dir = os.path.join(CWD, args.output_dir)

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir, ignore_errors=True)

    os.makedirs(args.output_dir)

    # create README.md file for the generated information about the models
    output_file = os.path.join(CWD, args.output_dir, README)
    command = 'touch {0}'.format(output_file)
    check_call(command, shell=True)

    """
        Determine Plot parameters for 32 channels
    """
    rows = 8
    cols = 4
    plots = rows * cols
    figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(60, 50))
    figure_loss, axes_loss = plt.subplots(nrows=rows, ncols=cols, figsize=(60, 50))
    r = 0
    c = 0

    # TODO: can add metadata to the README file about parameters chosen for models

    """
        Need to train and save a model for each channel in CHANNEL_CHOICES
    """
    for channel in CHANNEL_CHOICES:
        # Get all paths we want to read data from for classes: NONPD, PD medication, PD no medication
        all_data_paths = determine_data_paths(PATH_TO_DATASET, channel, int(args.size))

        data_set, labels = get_train_test_data(all_data_paths, int(args.image_size))

        # Normalize dataset
        data_set = data_set / 255.0
        # One-Hot encode labels
        labels = to_categorical(labels, CLASSES)

        print('-------------------------\n[INFO] Building Model\n-------------------------')

        # split training and test data
        (trainX, testX, trainY, testY) = train_test_split(data_set, labels, test_size=0.20, random_state=42)

        # building model
        model = Sequential()

        # Maybe pad images to not lose data along the border of the image
        # Maybe have a max pooling with 0 overlap. Meaning stride of (2, 2)
        # Maybe try Adadelta optimizer function
        # Maybe try mini batching

        # For evaluation I can use cross-validation

        # adding layers
        model.add(Conv2D(16, (3, 3), input_shape=(int(args.image_size), int(args.image_size), 3), activation=ACTIVATION, padding='same'))
        model.add(MaxPool2D(strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation=ACTIVATION, padding='same'))
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256, activation=ACTIVATION))

        model.add(Dense(2, activation=PREDICT_ACTIVATION))
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

        if print_metadata:
            print('-------------------------\n[INFO] Print Model Metadata\n-------------------------')
            print_model_metadata(output_file, model, args)
            print_metadata = False

        print('-------------------------\n[INFO] Train Model\n-------------------------')

        history = model.fit(trainX, trainY, epochs=int(args.epochs), validation_data=(testX, testY))

        print('-------------------------\n[INFO] Saving Model\n-------------------------')

        filename = os.path.join(args.output_dir, 'model.{0}'.format(channel))

        model.save(filename)

        print('-------------------------\n[INFO] Save model information to README.md\n-------------------------')
        output_str = ''

        output_str += '## Channel {0} Model\n\n'.format(channel)

        output_str += '| Epoch | Accuracy | Loss | Validation Accuracy | Validation Loss |\n'
        output_str += '| ---- | ------ | ------ | ------- | ------- |\n'
        i = 0
        while i < len(history.history['accuracy']):
            epoch = i + 1
            accuracy = history.history['accuracy'][i]
            loss = history.history['loss'][i]
            val_accuracy = history.history['val_accuracy'][i]
            val_loss = history.history['val_loss'][i]

            output_str += '| {0} | {1} | {2} | {3} | {4} |\n'.format(epoch, accuracy, loss, val_accuracy, val_loss)
            i += 1

        output_str += '\n\n'

        command = 'echo \'{0}\' >> {1}'.format(output_str, output_file)
        check_call(command, shell=True)

        print('-------------------------\n[INFO] Ploting Accuracy of Model\n-------------------------')
        axes[r, c].plot(history.history['accuracy'], linewidth=2.0)
        axes[r, c].plot(history.history['val_accuracy'], linewidth=2.0)
        axes[r, c].set_title(channel)

        print('-------------------------\n[INFO] Ploting Loss of Model\n-------------------------')
        axes_loss[r, c].plot(history.history['loss'], linewidth=2.0)
        axes_loss[r, c].plot(history.history['val_loss'], linewidth=2.0)
        axes_loss[r, c].set_title(channel)

        c += 1
        if c == 4:
            break
            c = 0
            r += 1


    accuracy_file = os.path.join(CWD, args.output_dir, ACCURACY_FILE)
    loss_file = os.path.join(CWD, args.output_dir, LOSS_FILE)
    for ax in axes.flat:
        ax.set(xlabel='Epoch', ylabel='Accuracy')

    for ax in axes_loss.flat:
        ax.set(xlabel='Epoch', ylabel='Loss')

    figure.tight_layout()
    figure_loss.tight_layout()
    figure.savefig(accuracy_file)
    figure_loss.savefig(loss_file)


if __name__ == '__main__':
    main()