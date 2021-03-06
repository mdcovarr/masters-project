"""
    Class used to load data in batchs, since data is too large to load into memory
"""
from tensorflow.keras.utils import Sequence
import numpy as np
from skimage.io import imread

class DataGenerator(Sequence):
    """
        Data Generator class to load data in batches
    """
    def __init__(self, image_filenames, labels, batch_size):
        """
            Default Constructor
        """
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        """
        Function to determine the number of batches to produce
        """
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """
            Function to get an item at a certain index
        """
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        data = np.array([imread(str(file_name)) for file_name in batch_x]) / 255.0
        # need to reshape data
        data_reshaped = data.reshape((data.shape[0], 32, 200, 200, 3))

        return data_reshaped, np.array(batch_y)

# Usage
# batch_size = 32
# my_train_batch_generator = DataGenerator(x_train_files, y_train, batch_size)
# my_validation_batch_generator = DataGenerator(x_val, y_val, batch_size)