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


def get_image_features_from_images(
    images: np.ndarray,
) -> np.ndarray:
    BINS = 256
    image_features = []
    for image in images:
        histogram_red = np.histogram(image[:, :, 0], bins=BINS, range=(0, 256))[0]
        histogram_green = np.histogram(image[:, :, 1], bins=BINS, range=(0, 256))[0]
        histogram_blue = np.histogram(image[:, :, 2], bins=BINS, range=(0, 256))[0]
        histogram = np.concatenate([histogram_red, histogram_green, histogram_blue])
        image_features.append(histogram)
    return np.array(image_features)


def get_images_from_directory_by_ids(
    directory: str,
    image_ids: np.ndarray,
    percent: float = 1.0,
) -> np.ndarray:
    images = []
    for image_id in image_ids:
        img_path = os.path.join(directory, image_id + ".png")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        images.append(img_np)
        if len(images) / len(image_ids) >= percent:
            break
    return np.array(images)


def load_images_from_directory(
    directory: str,
    percent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids, image_labels = get_image_ids_and_labels(directory)

    images = get_images_from_directory_by_ids(directory, image_ids, percent)

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

    image_features = get_image_features_from_images(np.array(images))
    image_features = preprocess_images(image_features)

    image_dataset = ImageDataset(
        images=image_features, labels=image_labels, ids=image_ids
    )

    return image_dataset


def load_test_images(percent: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_ids = np.loadtxt("test.csv", dtype=str)[1:]
    images = get_images_from_directory_by_ids("test", image_ids, percent)

    return np.stack(images), image_ids, None


def save_predictions_to_csv(
    predictions: np.ndarray, image_ids: np.ndarray, output_file: str = "submission.csv"
):
    with open(output_file, "w") as f:
        f.write("image_id,label\n")
        for image_id, prediction in zip(image_ids, predictions):
            f.write(f"{image_id},{prediction}\n")
    print(f"Predictions saved to {output_file}")
