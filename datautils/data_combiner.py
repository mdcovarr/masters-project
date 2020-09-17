"""
    Class to combine all data into a single directory

    This is in order to batch data
"""
import os
import glob
import shutil
from subprocess import check_call
from tensorflow.keras.utils import to_categorical
import numpy as np

class DataCombiner(object):
    """
        Class used to combine all data into a single directory
    """
    def __init__(self):
        """
            Default Constroctor
        """
        self.dest_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'all_ensemble_images')
        self.src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'stft-ensemble-200-40Hz')
        self.metadata_dest_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'ensemble_metadata')

    def run(self):
        """
            Function to initialze process of combining all data
        """
        # 1. make new destination dir
        if os.path.isdir(self.dest_dir):
            shutil.rmtree(self.dest_dir, ignore_errors=True)

        os.makedirs(self.dest_dir)

        patient_dirs = glob.glob(os.path.join(self.src_dir, '*'))

        for patient_dir in patient_dirs:
            # itertate through all directories of patients

            all_images = glob.glob(os.path.join(patient_dir, '*'))
            for image in all_images:
                # iterate through all directories of patients
                # 2. need to rename images
                patient_num = image.split(os.path.sep)[-2:-1][0]
                basename = os.path.basename(image)
                new_name = '{0}-{1}'.format(patient_num, basename)
                new_path = os.path.join(self.dest_dir, new_name)
                command = 'mv {0} {1}'.format(image, new_path)

                try:
                    # 3. need to place them in new directory
                    check_call(command, shell=True)
                except:
                    print('Error moving image {0}'.format(image))

    def save_metadata(self):
        """
            Function to get metadata infomation such as list of image names, and labels
        """
        labels = []

        # 1. make new destivation dir for metadata
        if os.path.isdir(self.metadata_dest_dir):
            shutil.rmtree(self.metadata_dest_dir, ignore_errors=True)

        os.makedirs(self.metadata_dest_dir)

        filenames = glob.glob(os.path.join(self.dest_dir, '*'))
        filenames.sort()

        for filename in filenames:
            if 'sub-hc' in filename:
                labels.append(0)
            elif 'sub-pd' in filename:
                labels.append(1)
            else:
                print('Error: Counld not determine label! exiting...')

        filenames = np.array(filenames)
        labels = np.array(labels)
        labels = to_categorical(labels)

        # need to save files
        np.save(os.path.join(self.metadata_dest_dir, 'data.npy'), filenames)
        np.save(os.path.join(self.metadata_dest_dir, 'labels.npy'), labels)

def main():
    """
        Start
    """
    data_combiner = DataCombiner()
    #data_combiner.run()
    data_combiner.save_metadata()

if __name__ == '__main__':
    main()