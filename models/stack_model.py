"""
    Module to create a stacked model from different channel models
"""
from tensorflow.keras.models import load_model
import os

CWD = os.path.dirname(os.path.realpath(__file__))
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

    for channel in model_channels:
        model_name = 'model.{0}'.format(channel)
        filename = os.path.join(CWD, '..', 'archive', 'models', 'stft_32Hz', model_name)
        model = load_model(filename)
        all_models.append(model)

def stacked_dataset(members, input_x):
    """
    Function used to get prediction of data for each model
    and stack results as input for the stacked model
    """

def main():
    """
    Script start
    """
    print('----------------- Loading all models... -------------------')
    members = load_all_models(SUFFIX_CHANNELS)

    print('----------------- All models loaded -------------------')

if __name__ == '__main__':
    main()
