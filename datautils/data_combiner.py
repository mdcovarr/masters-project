"""
    Class to combine all data into a single directory

    This is in order to batch data
"""
import os

class DataCombiner(object):
    """
        Class used to combine all data into a single directory
    """
    def __init__(self):
        """
            Default Constroctor
        """
        self.dest_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'all_images')
        self.src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'stft-ensemble-200-40Hz')

    def run(self):
        """
            Function to initialze process of combining all data
        """
        # 1. need to rename images
        # 2. need to place them in new directory
        pass
