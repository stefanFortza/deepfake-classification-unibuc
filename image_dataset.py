import numpy as np


class ImageDataset:
    def __init__(
        self, images: np.ndarray, labels: np.ndarray = None, ids: np.ndarray = None
    ):
        self.images = images
        self.labels = labels
        self.ids = ids

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
