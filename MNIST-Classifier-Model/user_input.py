import pygame
import torch
import numpy as np
import sys
from torchvision import transforms
from torch import nn

# Define the neural network model (same as before)
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

# Load the pre-trained model
model = SimpleNN()

# Load saved model state dict (don't retrain, just load the weights)
model.load_state_dict(torch.load(sys.argv[1]))  # Load the model file passed as argument
model.eval()  # Set the model to evaluation mode

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Start pygame
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
smallFont = pygame.font.SysFont('Arial', 20)
largeFont = pygame.font.SysFont('Arial', 40)

ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
classification = None

# Transform to normalize the image as the model expects
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

while True:

    # Check if game quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.fill(BLACK)

    # Check for mouse press
    click, _, _ = pygame.mouse.get_pressed()
    if click == 1:
        mouse = pygame.mouse.get_pos()
    else:
        mouse = None

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse and resetButton.collidepoint(mouse):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        classification = None

    # Generate classification
    if mouse and classifyButton.collidepoint(mouse):
        # Convert the drawn grid to a tensor, normalize it, and convert to float
        image_tensor = transform(np.array(handwriting).reshape(28, 28)).float()  # Convert to float
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
    # Make the prediction
        with torch.no_grad():
            output = model(image_tensor)
            classification = torch.argmax(output, dim=1).item()  # Get the predicted digit

    # Make the prediction
        with torch.no_grad():
            output = model(image_tensor)
            classification = torch.argmax(output, dim=1).item()  # Get the predicted digit
            # Make the prediction
            with torch.no_grad():
                output = model(image_tensor)
                classification = torch.argmax(output, dim=1).item()  # Get the predicted digit

    # Show classification if one exists
    if classification is not None:
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            100
        )
        screen.blit(classificationText, classificationRect)

    pygame.display.flip()
    