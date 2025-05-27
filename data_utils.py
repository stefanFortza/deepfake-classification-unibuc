from sklearn import preprocessing
import os
from PIL import Image
import numpy as np
from image_dataset import ImageDataset
import image_dataset


def get_image_ids_and_labels(directory: str) -> tuple[np.ndarray, np.ndarray]:
    ids_and_labels = np.loadtxt(directory + ".csv", delimiter=",", dtype=str)
    ids_and_labels = ids_and_labels[1:]

    ids = ids_and_labels[:, 0].astype(str)
    labels_as_int = ids_and_labels[:, 1].astype(int)

    return ids, labels_as_int


def load_images_from_directory(
    directory: str,
    percent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids, image_labels = get_image_ids_and_labels(directory)
    images: np.ndarray = []

    for image_id in image_ids:
        img_path = os.path.join(directory, image_id + ".png")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        images.append(img_np)
        if len(images) / len(image_ids) >= percent:
            break

    # images = images[: int(len(images) * percet)]
    image_labels = image_labels[: len(images)]
    image_ids = image_ids[: len(images)]

    return np.stack(images), image_ids, image_labels


def preprocess_images(images: np.ndarray) -> np.ndarray:
    def reshape_images(images: np.ndarray) -> np.ndarray:
        images = images.reshape((images.shape[0], -1))
        return images

    def normalize_images(images: np.ndarray) -> np.ndarray:
        return preprocessing.normalize(images, norm="l2")

    images = reshape_images(images)
    images = normalize_images(images)
    return images


def load_images(directory: str, percent: float = 1.0):
    if directory == "test":
        images, image_ids, image_labels = load_test_images(percent)
    else:
        images, image_ids, image_labels = load_images_from_directory(directory, percent)

    images = preprocess_images(images)

    image_dataset = ImageDataset(images=images, labels=image_labels, ids=image_ids)

    return image_dataset


def load_test_images(percent: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids = np.loadtxt("test.csv", dtype=str)[1:]
    images: np.ndarray = []

    for image_id in image_ids:
        img_path = os.path.join("test", image_id + ".png")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        images.append(img_np)
        if len(images) / len(image_ids) >= percent:
            break

    return np.stack(images), image_ids, None

    # images, image_ids, _ = load_images_from_directory(directory, percent)
    # images = preprocess_images(images)

    # image_dataset = ImageDataset(images=images, ids=image_ids)

    # return image_dataset


image_dataset = load_images("test", percent=1.0)
print(image_dataset)
