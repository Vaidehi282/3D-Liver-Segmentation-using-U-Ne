from glob import glob
import shutil
import os
import dicom2nifti
import nibabel as nib
import numpy as np

# Directories for input and output DICOM files (Images and Labels)
in_path_img = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom files/images'
out_path_img = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom_groups/images'
in_path_label = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom files/labels'
out_path_label = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom_groups/labels'

# Step 1: Group DICOM files for each patient into sets of 64
def group_dicom_files(input_path, output_path):
    """
    Organizes DICOM files into groups of 64 per patient.
    """
    for patient in sorted(glob(input_path + '/*')):
        patient_name = os.path.basename(patient)
        num_folders = int(len(glob(patient + '/*')) / 64)  # Calculate the number of groups
        
        for i in range(num_folders):
            output_path_name = os.path.join(output_path, f"{patient_name}_{i}")
            os.mkdir(output_path_name)
            print(f"Creating folder: {output_path_name}")
            
            for j, file in enumerate(sorted(glob(patient + '/*'))):
                if j == 64:
                    break  # Stop after moving 64 files
                shutil.move(file, output_path_name)

# Organize images and labels into groups of 64
group_dicom_files(in_path_img, out_path_img)
group_dicom_files(in_path_label, out_path_label)

# Directories for final NIfTI output
in_path_img = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom_groups/images/*'
in_path_lab = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/dicom_groups/labels/*'
out_path_img = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/niftifiles/images'
out_path_lab = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/niftifiles/labels'

list_images = glob(in_path_img)
list_labels = glob(in_path_lab)

# Step 2: Convert grouped DICOM files back to NIfTI format
def convert_dicom_to_nifti(dicom_list, output_path):
    """
    Converts DICOM series to NIfTI format.
    """
    for patient in dicom_list:
        patient_name = os.path.basename(patient)
        output_file = os.path.join(output_path, f"{patient_name}.nii.gz")
        print(f"Converting {patient} to {output_file}")
        dicom2nifti.dicom_series_to_nifti(patient, output_file)

# Convert grouped DICOM files to NIfTI format
convert_dicom_to_nifti(list_images, out_path_img)
convert_dicom_to_nifti(list_labels, out_path_lab)

# Step 3: Check if each NIfTI file contains labels
input_nifty_file = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/niftifiles/labels/*'
list_labels = sorted(glob(input_nifty_file))

for patient in list_labels:
    patient_name = os.path.basename(patient)  # Extract patient name without extension
    img = nib.load(patient)
    fdata = img.get_fdata()
    np_unique = np.unique(fdata)
    
    if len(np_unique) == 1:
        print(f"No labels found in: {patient}")

print("Processing complete! NIfTI files are saved.")
