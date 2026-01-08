from typing import List, Optional
import torch.nn as nn
import torch


class LinearClassifier(nn.Module):
    """Single linear classifier .
    This flattens the input image and applies a single `nn.Linear` layer
    to produce logits for `num_classes` outputs.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, C, H, W] -> flatten to [batch_size, input_dim]
        x = x.view(x.size(0), -1)
        return self.linear(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConv(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3, conv_filters: Optional[List[int]] = None, fc_units: Optional[List[int]] = None):
        super().__init__()
        if conv_filters is None: conv_filters = [32, 64, 128]
        if fc_units is None: fc_units = [64]

        layers = []
        ch = in_channels
        for out_ch in conv_filters:
            layers.append(ConvBlock(ch, out_ch))
            ch = out_ch
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4)) # Ensure fixed size before flatten

        fc_layers = []
        prev = ch * 4 * 4
        for units in fc_units:
            fc_layers.append(nn.Linear(prev, units))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(0.5))
            prev = units
        
        fc_layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SlimCNN(nn.Module):
    """Optimized slim architecture."""
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # Separable blocks
        self.blocks = nn.Sequential(
            SeparableConv(16, 32),
            nn.MaxPool2d(2),
            SeparableConv(32, 64),
            nn.MaxPool2d(2),
            SeparableConv(64, 128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x


def get_model(name: str, input_dim: int, num_classes: int, img_size: int = 128):
    """Function for returning a model instance for the given name."""
    name = name.lower()
    if name in ('logistic', 'logistic_regression', 'linear', 'linear_model', 'linear_regression'):
        return LinearClassifier(input_dim, num_classes)
    elif name == 'cnn':
        return ConfigurableCNN(num_classes=num_classes, in_channels=3)
    elif name == 'slim_cnn':
        return SlimCNN(num_classes=num_classes, in_channels=3)
    
    raise ValueError(f"Unknown model name: {name}. Supported: logistic, cnn, slim_cnn")
