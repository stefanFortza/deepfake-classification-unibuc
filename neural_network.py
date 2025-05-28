import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import ImageDataset

from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dim=3 * 256,
        hidden_dims=[512, 256],
        num_classes=5,
        dropout_rate=0.3,
        activation_fn=nn.ReLU,
    ):
        super().__init__()

        self.flatten = nn.Flatten()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.model = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization for layers followed by ReLU/LeakyReLU
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity=(
                        "relu"
                        if isinstance(self.model.modules, nn.ReLU)
                        else "leaky_relu"
                    ),
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.first_layer = nn.Linear(3 * 256, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(0.3)
#         self.second_layer = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout2 = nn.Dropout(0.3)
#         self.output_layer = nn.Linear(256, 5)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = F.relu(self.bn1(self.first_layer(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn2(self.second_layer(x)))
#         x = self.dropout2(x)
#         x = self.output_layer(x)
#         return x


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.first_layer = nn.Linear(3 * 256, 512)
#         self.second_layer = nn.Linear(512, 512)
#         self.output_layer = nn.Linear(512, 5)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = F.relu(self.first_layer(x))
#         x = F.relu(self.second_layer(x))
#         x = self.output_layer(x)
#         return x


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
