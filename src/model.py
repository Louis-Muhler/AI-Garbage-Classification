import torch.nn as nn
import torch.nn.functional as F

class SimpleGarbageCNN(nn.Module):
    """
    Vertieftes Modell ohne Batch Normalization.
    Nutzt Dropout (0.2) zur Regularisierung, um Overfitting bei 
    steigender Komplexität zu verhindern.
    """
    def __init__(self, num_classes, input_size=128):
        super(SimpleGarbageCNN, self).__init__()
        
        # Conv-Layer ohne BN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout auf 0.2 gesetzt
        self.dropout = nn.Dropout(0.2)
        
        # Berechnung: 4 Poolings -> 128 / 16 = 8
        self.final_spatial_size = input_size // 16 
        self.flattened_size = 256 * self.final_spatial_size * self.final_spatial_size
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Architektur: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, self.flattened_size)
        
        # Classifier mit Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x) # Optional: Dropout auch nach dem zweiten FC-Layer
        x = self.fc3(x)
        return x