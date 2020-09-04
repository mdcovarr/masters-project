"""
    Module used to stack the EEG readings of each channel in order to pass
    each data point to the ensemble model
"""

import os
import numpy as np
import glob
import argparse
import shutil
from PIL import Image

CWD = os.path.dirname(os.path.realpath(__file__))
LAST_IMAGE_COUNT = 45

def handle_arguments():
    """
    Function to set up commandline arguments
    :return args:
    """
    parser = argparse.ArgumentParser(description='Script used to combine image of each EEG channel into one.')

    parser.add_argument('-i', '--image-size', dest='image_size', required=True)
    parser.add_argument('-d', '--data-root', dest='data_root', required=True,
                        help='root of directory for training testing data images')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True,
            help='Flag used to determine the root output path to place images')
    args = parser.parse_args()

    return args

def get_patient_dirs(root_dir):
    """
    Function used to get the root director for all patients
    :param root_dir: root director of all image data
    :return patient_paths: list of all patient paths, one for each patient
    """
    search_path = os.path.join(root_dir, '[0-1]', '*')
    patient_paths = glob.glob(search_path)

    return patient_paths

def merge_all_channel_images(all_patient_paths, output_dir, image_resize):
    """
    Function used to merge all channel images into one
    :param all_patient_paths: list of all paths, one for each patient
    :param output_dir: output dir for new concatenated images
    :return:
    """
    # clean and make output directory
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    os.makedirs(output_dir)

    # need to iterate through all patients to combine image files
    for patient in all_patient_paths:
        patient_basename = os.path.basename(patient)
        channels_search = os.path.join(patient, '*')

        # make output dir for patient's new concatenated data
        patient_output_dir = os.path.join(output_dir, patient_basename)
        os.mkdir(patient_output_dir)

        # get image for each channel iterate chronologically i.e 0 ... 45
        for i in range(LAST_IMAGE_COUNT):
            image_list = []

            # get image for each channel
            image_name = '{0}.png'.format(i)
            images_search = os.path.join(channels_search, image_name)
            image_list = glob.glob(images_search)

            image_list.sort()

            # need to load all images and combine
            loaded_images = []
            for image in image_list:
                img = Image.open(image).convert('RGB').resize((image_resize, image_resize))
                loaded_images.append(img)

            # combine all loaded images
            # image_size = [width, height]
            image_size = loaded_images[0].size
            # total count of images
            images_count = len(loaded_images)

            new_image = Image.new('RGB',(image_size[0], images_count * image_size[1]), (250,250,250))

            for j in range(len(loaded_images)):
                new_image.paste(loaded_images[j],(0,j * image_size[1]))
                new_image.paste(loaded_images[1],(0, image_size[1]))

            image_output = os.path.join(patient_output_dir, '{0}.jpg'.format(i))
            new_image.save(image_output,"JPEG")

def main():
    """
        Start
    """
    # handle arguments
    args = handle_arguments()

    # get all patient paths
    data_root = os.path.join(CWD, '..', args.data_root)
    output_dir = os.path.join(CWD, '..', args.output_dir)
    all_patients = get_patient_dirs(data_root)

    merge_all_channel_images(all_patients, output_dir, int(args.image_size))


if '__main__' == __name__:
    main()
