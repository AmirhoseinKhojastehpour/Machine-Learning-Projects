import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Data transformation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Train dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the training dataset
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Define the neural network model with two layers
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # First layer with 784 input and 128 output neurons
        self.tanh = nn.Tanh()  # Tanh activation for hidden layer
        self.fc2 = nn.Linear(512, 10)  # Second layer with 128 input and 10 output neurons (for 10 classes)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.fc1(x)  # Apply first layer
        x = self.tanh(x)  # Apply tanh activation
        x = self.fc2(x)  # Apply second layer
        x = self.leakyRelu(x)  # Apply leakyRelu activation
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)  # Using Adam optimizer with learning rate of 0.001

# Define the number of epochs for training
num_epochs = 10  # Number of epochs

# Initialize lists to store loss and accuracy values
epoch_losses = []
epoch_accuracies = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update the model's weights

        running_loss += loss.item()

    # Calculate average loss and accuracy for this epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total

    # Store the results
    epoch_losses.append(avg_loss)
    epoch_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Plot Loss and Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot loss
ax1.plot(range(1, num_epochs+1), epoch_losses, label='Loss')
ax1.set_title('Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy
ax2.plot(range(1, num_epochs+1), epoch_accuracies, label='Accuracy', color='orange')
ax2.set_title('Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()

# Set the model to evaluation mode
model.eval()

# Visualize some results
num_images = 10  # Number of images to display
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Load the test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get some test images (using a subset from the train_loader here)
data_iter = iter(test_loader)
images, labels = next(data_iter)

# Make predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Display the images and predictions
for i in range(num_images):
    ax = axes[i]
    ax.imshow(images[i].squeeze(), cmap='gray')
    ax.set_title(f'True: {labels[i]} Pred: {predicted[i]}')
    ax.axis('off')

# Adjust layout to add more space between the images
plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()

# Initialize an array to store the number of correct predictions per class
class_correct = [0] * 10
class_total = [0] * 10

# Set the model to evaluation mode
model.eval()

# Loop through the test data
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        # Update correct and total counts for each class
        for i in range(len(labels)):
            label = labels[i]
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

# Calculate accuracy for each class
class_accuracies = [100 * correct / total if total > 0 else 0 for correct, total in zip(class_correct, class_total)]

# Plot the accuracy for each class
plt.bar(range(10), class_accuracies, color='blue')
plt.title('Accuracy per Class')
plt.xlabel('Digit')
plt.ylabel('Accuracy (%)')
plt.xticks(range(10))
plt.show()

# Print out the accuracy for each class
for i in range(10):
    print(f'Class {i}: {class_accuracies[i]:.2f}%')

# Evaluate on the test set
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the Model
# torch.save(model.state_dict(), 'model.pth')
