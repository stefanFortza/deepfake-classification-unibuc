import torch
import torch.nn as nn
from neural_network import NeuralNetwork # Ensure NeuralNetwork is imported
from data_utils import load_images # Ensure load_images is imported
from neural_network import get_dataloader_from_image_dataset # Ensure get_dataloader_from_image_dataset is imported
import matplotlib.pyplot as plt

# Define the learning rates to test
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = {} # To store accuracy for each learning rate

NUM_EPOCHS_LR_TEST = 10 # You can adjust this if needed for faster experimentation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assuming train_loader, validation_loader are already defined from previous cells
# If not, you might need to redefine them or ensure they are in scope.
# For this example, I'll assume they are available.
# train_dataset = load_images("train", percent=1)
# validation_dataset = load_images("validation")
# train_loader = get_dataloader_from_image_dataset(train_dataset, batch_size=64)
# validation_loader = get_dataloader_from_image_dataset(validation_dataset, batch_size=64)


for lr in learning_rates:
    print(f"\n--- Training with Learning Rate: {lr} ---")
    
    # 1. Re-initialize the Model and Optimizer
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    epoch_losses_lr_test = []
    
    # 2. Training Loop
    model.train(True)
    for epoch in range(NUM_EPOCHS_LR_TEST):
        running_loss = 0.0
        num_batches = 0
        for batch_idx, (image_batch, labels_batch) in enumerate(train_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            image_batch = image_batch.permute(0, 3, 1, 2)

            predictions = model(image_batch)
            loss = loss_fn(predictions, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
        
        epoch_loss = running_loss / num_batches
        epoch_losses_lr_test.append(epoch_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS_LR_TEST}, LR: {lr}, Avg Loss: {epoch_loss:.4f}")

    # 3. Validation Loop
    model.eval()
    correct = 0.0
    total_loss_val = 0.0
    num_val_batches = 0
    
    # Ensure validation_loader is defined and accessible
    # It should be defined in a previous cell (e.g., cell id b8071e3a)
    size_val = len(validation_loader.dataset)

    with torch.no_grad():
        for image_batch, labels_batch in validation_loader:
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            image_batch = image_batch.permute(0, 3, 1, 2)
            
            pred = model(image_batch)
            total_loss_val += loss_fn(pred, labels_batch).item()
            correct += (pred.argmax(1) == labels_batch).type(torch.float).sum().item()
            num_val_batches +=1
            
    accuracy = 100 * correct / size_val
    avg_val_loss = total_loss_val / num_val_batches # Or size_val if you want per-sample loss
    
    print(f"LR: {lr} -> Validation Accuracy: {accuracy:.2f}%, Avg Validation Loss: {avg_val_loss:.4f}")
    results[lr] = accuracy

# 4. Print Results
print("\n--- Learning Rate vs. Accuracy ---")
for lr, acc in results.items():
    print(f"Learning Rate: {lr}, Validation Accuracy: {acc:.2f}%")

# Optional: Plot results
lrs_plot = list(results.keys())
accs_plot = list(results.values())

plt.figure(figsize=(10, 5))
plt.plot([str(lr) for lr in lrs_plot], accs_plot, marker='o')
plt.title('Learning Rate vs. Validation Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (%)')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, marker='o', linestyle='-')
plt.title('Epoch vs. Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.xticks(range(1, NUM_EPOCHS + 1))
plt.grid(True)
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


cm = confusion_matrix(validation_dataset.labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for CNN")
plt.show()