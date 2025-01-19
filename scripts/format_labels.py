##
# Formats the labels so they use np.int64 data type (required by nnUNetv2)

import nibabel as nib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def fix_labels(label_file_path):
    # Load the NIfTI file
    img = nib.load(label_file_path)
    data = img.get_fdata()

    # Round the values to integers
    data = np.round(data).astype(np.int64)

    # Create a new NIfTI image with the corrected data
    fixed_img = nib.Nifti1Image(data, img.affine, header=img.header)

    # Save the corrected NIfTI file back
    nib.save(fixed_img, label_file_path)
    print(f"Fixed and saved: {label_file_path}")

def fix_labels_in_directory(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".nii.gz"):
            label_file_path = os.path.join(directory, filename)
            fix_labels(label_file_path)


if __name__ == "__main__":
    # Path to the directory containing your label files (e.g., labelsTr)
    labels_directory = os.path.join(os.getenv("nnUNet_raw"), f"Dataset"+os.getenv("DATASET_ID")+"_"+os.getenv("DATASET_NAME"), "labelsTr")
    fix_labels_in_directory(labels_directory)
