import torch.nn as nn
import torch.nn.functional as F

class SimpleGarbageCNN(nn.Module):
    """
    A basic Convolutional Neural Network (CNN) for garbage classification.
    This serves as a baseline model without advanced optimizations.
    """
    def __init__(self, num_classes):
        super(SimpleGarbageCNN, self).__init__()
        
        # Convolutional Layer 1: Input 3 channels (RGB), Output 16 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Convolutional Layer 2: Output 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Convolutional Layer 3: Output 64 filters
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max Pooling Layer: Reduces spatial dimensions by half (224 -> 112 -> 56 -> 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer 1: 
        # After 3 pooling layers, a 224x224 image becomes 28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        
        # Final Output Layer: One node per garbage category
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply first convolution, activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolution, activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolution, activation, and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps into a vector
        x = x.view(-1, 64 * 28 * 28)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Output layer (Logits)
        x = self.fc2(x)
        
        return x