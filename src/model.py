import torch.nn as nn
import torch.nn.functional as F

class SimpleGarbageCNN(nn.Module):
    """
    Komplexeres Modell mit Batch Normalization, mehr Layern und mehr Neuronen.
    Ziel: Underfitting vermeiden durch höhere Kapazität.
    Struktur: 5 Conv-Blöcke (bis 512 Filter), BN, Dropout.
    """
    def __init__(self, num_classes, input_size=224):
        super(SimpleGarbageCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Block 5 (Neu: Tieferes Netz)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.3)
        
        # Berechnung: 5 Poolings -> input_size / 32
        self.final_spatial_size = input_size // 32 
        # Falls input_size=224 -> 7x7. Flattened size: 512 channels * 7 * 7
        self.flattened_size = 512 * self.final_spatial_size * self.final_spatial_size
        
        # Größere FC Layer
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = x.view(-1, self.flattened_size)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x