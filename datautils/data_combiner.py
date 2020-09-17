"""
    Class to combine all data into a single directory

    This is in order to batch data
"""
import os
import glob
import shutil
from subprocess import check_call

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


def main():
    """
        Start
    """
    data_combiner = DataCombiner()
    data_combiner.run()

if __name__ == '__main__':
    main()