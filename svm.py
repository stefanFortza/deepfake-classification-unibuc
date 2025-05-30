from data_utils import load_images, ImageDataset, save_predictions_to_csv
from sklearn import svm
import matplotlib.pyplot as plt

# Load the data
train_dir = "train"
train_image_dataset: ImageDataset = load_images(train_dir)
validation_image_dataset: ImageDataset = load_images("validation")
test_image_dataset = load_images("test")
print(f"Loaded {len(train_image_dataset)} images from {train_dir} directory.")
print(f"Loaded {len(validation_image_dataset)} images from validation directory.")
print(f"Loaded {len(test_image_dataset)} images from test directory.")

# Model training and validation
print(train_image_dataset.images.shape)
model = svm.SVC()
model.fit(train_image_dataset.images, train_image_dataset.labels)


# Predictions on test set
predictions = model.predict(test_image_dataset.images)
print(f"Predictions for {predictions} images in test dataset.")

# Save predictions
save_predictions_to_csv(predictions, test_image_dataset.ids, "predictions.csv")
