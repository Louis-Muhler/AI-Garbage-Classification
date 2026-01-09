import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block to help with the 'Vanishing Gradient Problem' in deep networks.
    Input -> [Conv-BN-ReLU-Conv-BN] + [Shortcut] -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Adjust shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_block(x)
        out += self.shortcut(x) # Skip Connection: Add input to output
        out = F.relu(out)
        return out

class SimpleGarbageCNN(nn.Module):
    """
    Improved ResNet-Style CNN.
    Residual connections help deep networks learn efficiently.
    """
    def __init__(self, num_classes, input_size=None):
        super(SimpleGarbageCNN, self).__init__()
        
        # Initial Convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # ResNet Blocks + Pooling
        self.block1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.block3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.block4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.block5 = ResidualBlock(512, 512)
        # No 5th pooling here, GAP handles the rest to keep more spatial info
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3), # Dropout slightly increased for ResNet
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = self.block5(x) # Last block without pooling
        
        x = self.global_pool(x)
        x = self.fc(x)
        return x