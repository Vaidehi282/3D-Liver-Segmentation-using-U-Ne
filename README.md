# 3D Liver Segmentation using U-Net

This repository provides a complete pipeline for **3D liver segmentation using U-Net**. The pipeline includes the following steps:

## 1. Convert NIfTI to DICOM
The first step is converting NIfTI (.nii.gz) files to DICOM format using the **Slicer app**, which prepares the data for further processing.

## 2. DICOM File Handling
- DICOM files are grouped into batches of 64 per patient.
- These grouped files are moved into separate directories to maintain organized storage.
- The grouped DICOM files are then converted back into NIfTI format for processing.

## 3. Dataset
The dataset used for this project is sourced from the **Medical Decathlon** challenge, specifically for liver segmentation tasks. The dataset contains various 3D medical imaging data, including liver scans, from a total of **130 patients**. However, for training purposes, only **13 patients** were used, achieving a **Dice score of 0.69**. The score is expected to improve with the inclusion of more data.

## 4. Preprocessing
The data is loaded and preprocessed using MONAI, including transformations such as rescaling, cropping, and ensuring the correct channel format. The preprocessing pipeline includes:

- **Data Preparation**: Loading image and label file paths.
- **Transformation Pipelines**: Separate pipelines for training and validation data. The transformations include:
  - Rescaling and cropping of the images
  - Ensuring the correct channel format
  - Resizing the images and labels to the dimensions (128x128x64)

## 5. U-Net Model
The model used for liver segmentation is a **3D U-Net**. The setup includes:
- **Model Architecture**: 3D U-Net with batch normalization layers for stability.
- **Loss Function**: The Dice Loss function, optimized for segmentation tasks, minimizing errors between predicted and ground truth labels.
- **Training**: The model is trained using an Adam optimizer with a learning rate of 1e-5 and weight decay for regularization. Training proceeds for up to 100 epochs, with periodic saving of model checkpoints.

## 6. Final Output
The final output of the model consists of segmented liver regions, which can be used for medical imaging applications like diagnosis and treatment planning.

---

This repository automates the entire process from converting data to DICOM format, preprocessing, training the U-Net model, and obtaining segmentation results.
