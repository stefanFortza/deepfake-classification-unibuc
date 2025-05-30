import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import ImageDataset

from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the relu function
        self.relu_fn = nn.ReLU()

        # Define flannten function
        self.flatten = nn.Flatten()

        # Define dropout function
        self.dropout_fn = nn.Dropout(0.5)

        # Image has shape: 3 x 100 x 100

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1_conv = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 50 x 50

        # Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2_conv = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64 x 25 x 25

        # Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3_conv = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 x 12 x 12

        # Neural Network Layers (Fully Connected Layers)
        # The input is the output of the last convolutional layer
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.bn1_fc = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2_fc = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(256, 5)  # Assuming 5 output classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu_fn(self.bn1_conv(self.conv1(x))))
        x = self.pool2(self.relu_fn(self.bn2_conv(self.conv2(x))))
        x = self.pool3(self.relu_fn(self.bn3_conv(self.conv3(x))))

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.dropout_fn(self.relu_fn(self.bn1_fc(self.fc1(x))))
        x = self.dropout_fn(self.relu_fn(self.bn2_fc(self.fc2(x))))
        x = self.output_layer(x)
        return x


# Creates a dataloader based on the image_dataset
def get_dataloader_from_image_dataset(
    image_dataset: ImageDataset, batch_size: int, is_test: bool = False
):
    images_tensor = image_dataset.get_images_tensor()
    labels_tensor = image_dataset.get_labels_tensor()

    dataset = TensorDataset(images_tensor, labels_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if not is_test else False,
        # num_workers=4,
        # pin_memory=True,
    )
