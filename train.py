from monai.networks.nets import UNet
from monai.networks.layers import  Norm
from monai.losses import DiceLoss

import torch
from utilities import train
from preprocess_func import prepare

# Define directories for input data and model storage
data_dir = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/niftifiles'
model_dir = '/Users/vaidehitaraklad/Machine Learning/Jupyter/Projects/Liver Segmentation/results'

# Prepare the dataset for training (likely includes loading and preprocessing)
data_in = prepare(data_dir)

# Set the device for training (CPU by default, change to GPU if available)
device = torch.device("cpu")  
print(f"Using device: {device}")


model = UNet(
    spatial_dims = 3,   # doing segmentation of 3d therefore 3 dimensions
    in_channels = 1,    # Single-channel input (grayscale images)
    out_channels = 2,   # Two output classes (background & foreground)
    channels = (16, 32, 64, 128, 256), # encoder-decoder feature maps
    strides = (2,2,2,2),
    num_res_units = 2,
    norm = Norm.BATCH,  # Batch normalization for stability
).to(device)            # Move model to the specified device (CPU/GPU)

# Define the Dice loss function for segmentation tasks
# - `to_onehot_y=True` ensures labels are converted to one-hot encoding
# - `sigmoid=True` applies a sigmoid activation function to the outputs
# - `squared_pred=True` squares predictions, making small segmentation errors more penalized
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)

# Define the optimizer (Adam) with:
# - Learning rate: 1e-5 (low to ensure stable convergence)
# - Weight decay: 1e-5 (regularization to prevent overfitting)
# - AMSGrad: True (variation of Adam to improve convergence)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

# Train the model using the `train` function
# Parameters:
# - `model`: The U-Net model
# - `data_in`: The prepared training data (train & validation loaders)
# - `loss_function`: The Dice loss function
# - `optimizer`: Adam optimizer for updating model weights
# - `100`: Maximum number of training epochs
# - `model_dir`: Directory to save model checkpoints and logs
train(model, data_in, loss_function, optimizer, 100, model_dir)
