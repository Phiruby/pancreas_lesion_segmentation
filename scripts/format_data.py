#######################################################################################
# This file is used to format the data in the data folder 
# to fit nnUNet's requirements for raw data. See
# https://github.com/DIAGNijmegen/nnUNet_v2/blob/master/documentation/dataset_format.md
#######################################################################################

import os
from dotenv import load_dotenv
import shutil

load_dotenv()

def create_directory(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        print("Creating directory: ", directory)
        os.makedirs(directory)

def get_files(data_dir):
    """
    Get all files in the data directory.
    (Recursively)
    
    Args:
        data_dir (str): The path of the data directory.
    
    Returns:
        list: A list of all files in the data directory.
    """
    #recursively list all files within
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def filter_images_and_labels(files):
    """
    Filter images and labels from the list of files.
    
    Args:
        files (list): A list of files.
    
    Returns:
        tuple: A tuple containing a list of images and a list of labels.
    """
    # this is how the images are labelled 
    images = [f for f in files if f.endswith('0000.nii.gz')]

    # remaining items are labels
    labels = [f for f in files if f not in images]
    return images, labels

def move_files(files, dest_dir):
    """
    Move files from the source directory to the destination directory.
    
    Args:
        files (list): A list of files to move.
        src_dir (str): The source directory.
        dest_dir (str): The destination directory.
    """

    for file in files:
        filename = os.path.basename(file)  # Extract the filename
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(file, dest_path)

def format_data(data_dir, output_dir):
    """
    Format the data to fit nnUNet's requirements for raw data.
    
    Args:
        data_dir (str): The path of the data directory.
        output_dir (str): The path of the output directory.
    """
    create_directory(output_dir)

    subdirs = ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs', 'imagesVal', 'labelsVal']
    for subdir in subdirs:
        create_directory(os.path.join(output_dir, subdir))

    # go through each (train, test, val) in the provided data
    for subset in ['train', 'test', 'val']:
        subset_dir = os.path.join(data_dir, subset)
        if os.path.exists(subset_dir):
            subset_files = get_files(subset_dir)
            subset_images, subset_labels = filter_images_and_labels(subset_files)
            if subset == 'train':
                move_files(subset_images, os.path.join(output_dir, 'imagesTr'))
                move_files(subset_labels, os.path.join(output_dir, 'labelsTr'))
            elif subset == 'test':
                move_files(subset_images, os.path.join(output_dir, 'imagesTs'))
                move_files(subset_labels, os.path.join(output_dir, 'labelsTs'))
            elif subset == 'val':
                move_files(subset_images, os.path.join(output_dir, 'imagesVal'))
                move_files(subset_labels, os.path.join(output_dir, 'labelsVal'))

if __name__ == '__main__':
    # Normalize paths from .env variables
    nnUNet_raw = os.path.normpath(os.getenv("nnUNet_raw"))
    dataset_id = os.getenv("DATASET_ID")
    dataset_name = os.getenv("DATASET_NAME")
    data_dir = os.path.normpath(os.getenv("DATA_DIR"))
    
    output_dir = os.path.join(nnUNet_raw, f"Dataset{dataset_id}_{dataset_name}")
    format_data(data_dir, output_dir)
    