from data_utils import save_predictions_to_csv
from neural_network import NeuralNetwork
from data_utils import load_images
from neural_network import get_dataloader_from_image_dataset
import torch

# We load the data
train_dataset, scaler = load_images("train", percent=1)
test_dataset, _ = load_images("test", percent=1, scaler=scaler)
validation_dataset, _ = load_images("validation", percent=1, scaler=scaler)


# We get the data in the form of a PyTorch dataset
train_loader = get_dataloader_from_image_dataset(train_dataset, batch_size=64)
test_loader = get_dataloader_from_image_dataset(
    test_dataset, batch_size=64, is_test=True
)
validation_loader = get_dataloader_from_image_dataset(
    validation_dataset, batch_size=64, is_test=True
)


# We create the NeuralNetwork, optimizer, loss function and define the parameters
model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

NUM_EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

# We train the model

epoch_losses = []
model.train(True)
for i in range(NUM_EPOCHS):
    print(f"Starting epoch {i+1}/{NUM_EPOCHS}")
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (image_batch, labels_batch) in enumerate(train_loader):
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)

        # Change shape from (batch_size, 100, 100, 3) to (batch_size, 3, 100, 100)
        # 3 matrices of 100*100 are used as the CNN operates on each color channel separately
        image_batch = image_batch.permute(0, 3, 1, 2)

        predictions = model(image_batch)
        loss = loss_fn(predictions, labels_batch)

        running_loss += loss.item()
        num_batches += 1

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {i+1}/{NUM_EPOCHS}, Batch {batch_idx}, Loss: {loss.item()}")

    epoch_loss = running_loss / num_batches
    epoch_losses.append(epoch_loss)

# Validation dataset accuracy

correct = 0.0
test_loss = 0.0
size = len(validation_loader.dataset)
model.to(device)
model.eval()
predictions = []
with torch.no_grad():
    for image_batch, labels_batch in validation_loader:
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)

        # Change shape from (batch_size, 100, 100, 3) to (batch_size, 3, 100, 100)
        # 3 matrices of 100*100 are used as the CNN operates on each color channel separately
        image_batch = image_batch.permute(0, 3, 1, 2)

        # Calculate the predictions for the validation dataset
        pred = model(image_batch)

        predictions.append(pred)

        test_loss += loss_fn(pred, labels_batch).item()

        correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()

correct /= size
test_loss /= size
print(f"Accuracy: {(100*correct):>0.1f}%, Loss: {test_loss:>8f} \n")

# As we calculate the predictions on each batch we need to reduce one dimension
predictions = torch.cat(predictions, dim=0)
predictions = predictions.argmax(dim=1).cpu().numpy()
print(predictions.shape)

# Test dataset predictions

correct = 0.0
test_loss = 0.0
size = len(validation_loader.dataset)
model.to(device)
model.eval()
predictions = []
with torch.no_grad():
    for image_batch, labels_batch in test_loader:
        image_batch = image_batch.to(device)

        # Change shape from (batch_size, 100, 100, 3) to (batch_size, 3, 100, 100)
        # 3 matrices of 100*100 are used as the CNN operates on each color channel separately
        image_batch = image_batch.permute(0, 3, 1, 2)

        # Calculate the predictions for the test dataset
        pred = model(image_batch)

        predictions.append(pred)

# As we calculate the predictions on each batch we need to reduce one dimension
predictions = torch.cat(predictions, dim=0)
predictions = predictions.argmax(dim=1).cpu().numpy()
print(predictions)


# We save the predictions to a CSV file
ids = test_dataset.ids

save_predictions_to_csv(predictions, ids, "predictions2.csv")
