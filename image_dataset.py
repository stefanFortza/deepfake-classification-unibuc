import numpy as np
import torch


class ImageDataset:
    def __init__(
        self, images: np.ndarray, labels: np.ndarray = None, ids: np.ndarray = None
    ):
        self.images = images
        self.labels = labels
        self.ids = ids

    def get_image_by_id(self, image_id: str):
        if self.ids is None:
            raise ValueError("Image IDs are not available in this dataset.")

        idx = np.where(self.ids == image_id)[0]
        if idx.size == 0:
            raise ValueError(f"Image ID {image_id} not found in the dataset.")
        return self.images[idx[0]], (
            self.labels[idx[0]] if self.labels is not None else None
        )

    def get_images_tensor(self):
        return torch.tensor(self.images, dtype=torch.float32)

    def get_labels_tensor(self):
        if self.labels is not None:
            return torch.tensor(self.labels, dtype=torch.long)
        else:
            return torch.zeros((self.images.shape[0],), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.labels is not None and self.ids is not None:
            return self.images[idx], self.labels[idx], self.ids[idx]
        elif self.labels is not None:
            return self.images[idx], self.labels[idx]
        elif self.ids is not None:
            return self.images[idx], self.ids[idx]
        else:
            return self.images[idx]

    def __repr__(self):
        return f"ImageDataset(images={self.images.shape}, labels={self.labels.shape if self.labels is not None else None}, ids={self.ids.shape if self.ids is not None else None})"
