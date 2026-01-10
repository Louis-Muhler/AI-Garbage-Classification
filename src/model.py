from typing import List, Optional, Any, cast
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------
# 1. Helper Blocks
# ----------------------------

class LinearClassifier(nn.Module):
    """
    Single linear classifier.
    Flattens the input and applies a single Linear layer.
    Useful for baseline benchmarks (Logistic Regression).
    """
    def __init__(self, input_dim: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, C, H, W] -> flatten to [batch_size, input_dim]
        x = x.view(x.size(0), -1)
        return self.linear(x)


class SeparableConv(nn.Module):
    """
    Depthwise separable convolution for efficient parameter use.
    Used in SlimCNN.
    """
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


class ResidualBlock(nn.Module):
    """
    Residual Block for the Custom ResNet (improved architecture).
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
        out += self.shortcut(x) # Skip Connection
        out = F.relu(out)
        return out


# ----------------------------
# 2. Architectures
# ----------------------------

class SimpleCNN(nn.Module):
    """
    Optimized lightweight architecture using Separable Convolutions.
    """
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


class CustomResNet(nn.Module):
    """
    Improved ResNet-Style CNN.
    Uses Residual Connections and deeper layers for better accuracy.
    """
    def __init__(self, num_classes, input_size=None):
        super(CustomResNet, self).__init__()
        
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
            nn.Dropout(0.3),
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
        return self.fc(x)


# ----------------------------
# 3. Model Factory
# ----------------------------

def build_transfer_model(num_classes: int, model_name: str = "resnet18", dropout: float = 0.5, freeze_backbone: bool = False) -> nn.Module:
    """Build a pre-trained model for transfer learning."""
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown transfer model: {model_name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head for specific models
    if model_name.startswith("resnet"):
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            fc_layer = cast(nn.Linear, model.fc)
            num_ftrs = fc_layer.in_features
            model_any = model # type: Any
            model_any.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )

    elif model_name.startswith("efficientnet"):
        classifier = getattr(model, 'classifier', None)
        if isinstance(classifier, nn.Sequential) and len(classifier) > 1 and isinstance(classifier[1], nn.Linear):
             lin_layer = cast(nn.Linear, classifier[1])
             num_ftrs = lin_layer.in_features
             model_any = model # type: Any
             model_any.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )

    elif model_name == "mobilenet_v3_large":
        # MobileNetV3 classifier structure: [0]Linear -> [1]Hardswish -> [2]Dropout -> [3]Linear
        # We replace the last Linear layer and update defaults
        classifier = getattr(model, 'classifier', None)
        if isinstance(classifier, nn.Sequential):
            # The last layer is the classification head
            last_layer_idx = len(classifier) - 1
            if isinstance(classifier[last_layer_idx], nn.Linear):
                in_features = classifier[last_layer_idx].in_features
                # Modify the Dropout layer before it if it exists
                if last_layer_idx > 0 and isinstance(classifier[last_layer_idx-1], nn.Dropout):
                    classifier[last_layer_idx-1] = nn.Dropout(p=dropout)
                
                # Replace the last Linear layer
                classifier[last_layer_idx] = nn.Linear(in_features, num_classes)

    return model


def get_model(name: str, input_dim: int, num_classes: int, img_size: int = 128, **kwargs):
    """
    Factory function to retrieve model instances by name.
    """
    name = name.lower()
    if name in ('logistic', 'linear'):
        return LinearClassifier(input_dim, num_classes)
    elif name == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes, in_channels=3)
    elif name == 'custom_resnet':
        return CustomResNet(num_classes=num_classes)
    elif name in ['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_large']:
        return build_transfer_model(num_classes=num_classes, model_name=name, **kwargs)

    raise ValueError(f"Unknown model name: {name}. Supported: logistic, simple_cnn, custom_resnet, resnet18, resnet50, efficientnet_b0, mobilenet_v3_large")
