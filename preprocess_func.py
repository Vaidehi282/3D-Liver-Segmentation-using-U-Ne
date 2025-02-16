import pandas as pd
import numpy as np
import os
from glob import glob
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized, 
    Orientationd
)
from monai.data import DataLoader, Dataset, DataLoader
from monai.utils import first
import matplotlib.pyplot as plt


def prepare(data_dir):
    """
    Prepares the dataset by loading image and label file paths,
    applying preprocessing transformations, and creating DataLoaders.
    """

    train_images = sorted(glob(os.path.join(data_dir, 'images/*')))
    train_labels = sorted(glob(os.path.join(data_dir,'labels/*')))
    test_images = sorted(glob(os.path.join(data_dir, 'images_test/*')))
    test_labels = sorted(glob(os.path.join(data_dir, 'labels_test/*')))

    # Step 2: Create a list of dictionaries for training and validation datasets
    train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    val_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]
    # print(train_files)

    # Define the transformation pipeline for training data 
    train_transform = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 1.0), mode=('bilinear', 'nearest')),
            ScaleIntensityRanged(keys=['image'], a_min=-200, a_max = 200, b_min = 0.0, b_max = 1.0, clip = True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    # Define the transformation pipeline for validation data
    val_transform = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 1.0), mode=('bilinear', 'nearest')),
            ScaleIntensityRanged(keys=['image'], a_min=-200, a_max = 200, b_min = 0.0 , b_max = 1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    # creating dataset and applying mentioned transforms automatically
    train_ds = Dataset(data = train_files, transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size = 1)

    val_ds = Dataset(data = val_files, transform = val_transform)
    val_loader = DataLoader(val_ds, batch_size = 1)

    return train_loader, val_loader