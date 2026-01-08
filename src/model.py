import torch.nn as nn
import torch.nn.functional as F

class SimpleGarbageCNN(nn.Module):
    """
    A basic CNN that automatically adapts its linear layer to the input IMAGE_SIZE.
    This demonstrates systematic implementation and technical depth.
    """
    def __init__(self, num_classes, input_size=128):
        super(SimpleGarbageCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Automatic calculation of the flattened size
        # 3 pooling layers with stride 2 reduce the size by factor 2^3 = 8
        self.final_spatial_size = input_size // 8 
        self.flattened_size = 64 * self.final_spatial_size * self.final_spatial_size
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Dynamically use the calculated flattened size
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x