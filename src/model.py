import torch.nn as nn
import torch.nn.functional as F

class SimpleGarbageCNN(nn.Module):
    """
    Verbreiterte Version des CNNs. 
    Filter-Setup: 32 -> 64 -> 128 (statt vorher 16 -> 32 -> 64).
    Erhöht die Kapazität für komplexere Müll-Merkmale.
    """
    def __init__(self, num_classes, input_size=128):
        super(SimpleGarbageCNN, self).__init__()
        
        # Schicht 1: Erhöht auf 32 Filter
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Schicht 2: Erhöht auf 64 Filter
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Schicht 3: Erhöht auf 128 Filter
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Berechnung der Flattened-Größe bleibt logisch gleich:
        # 3 Pooling-Layer reduzieren die Auflösung um Faktor 2^3 = 8
        self.final_spatial_size = input_size // 8 
        # ACHTUNG: Hier muss nun die neue Filteranzahl (128) stehen!
        self.flattened_size = 128 * self.final_spatial_size * self.final_spatial_size
        
        # Auch die vollvernetzte Schicht (FC) verbreitern wir für die 128 Merkmale
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Dynamischer View auf die neue flattened_size
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x