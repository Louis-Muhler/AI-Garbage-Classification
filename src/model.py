from typing import List, Optional, Any, cast
import torch.nn as nn
import torch
import torchvision.models as models


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


def build_transfer_model(num_classes: int, model_name: str = "resnet18", dropout: float = 0.5, freeze_backbone: bool = True) -> nn.Module:
    """Build a pre-trained model for transfer learning.
    
    Supported: resnet18, resnet50, efficientnet_b0
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown transfer model: {model_name}")

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier head
    if model_name.startswith("resnet"):
        # Explicit type cast or assumption that model has fc attribute of type Linear
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            # Use cast to strictly tell the type checker this is a Linear layer
            fc_layer = cast(nn.Linear, model.fc)
            num_ftrs = fc_layer.in_features
            
            # Helper for assignment to avoid linter errors with torchvision models
            model_any = model # type: Any
            model_any.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )
            
    elif model_name.startswith("efficientnet"):
        # For efficientnet_b0, classifier[1] is the Linear layer.
        classifier = getattr(model, 'classifier', None)
        if isinstance(classifier, nn.Sequential) and len(classifier) > 1 and isinstance(classifier[1], nn.Linear):
             # Use cast for the linear layer inside sequential
             lin_layer = cast(nn.Linear, classifier[1])
             num_ftrs = lin_layer.in_features
             
             model_any = model # type: Any
             model_any.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, num_classes)
            )

    return model


def get_model(name: str, input_dim: int, num_classes: int, img_size: int = 128, **kwargs):
    """Function for returning a model instance for the given name."""
    name = name.lower()
    if name in ('logistic', 'logistic_regression', 'linear', 'linear_model', 'linear_regression'):
        return LinearClassifier(input_dim, num_classes)
    elif name == 'cnn':
        return ConfigurableCNN(num_classes=num_classes, in_channels=3)
    elif name == 'slim_cnn' or name == 'slim':
        return SlimCNN(num_classes=num_classes, in_channels=3)
    elif name in ['resnet18', 'resnet50', 'efficientnet_b0']:
        return build_transfer_model(num_classes=num_classes, model_name=name, **kwargs)
    
    raise ValueError(f"Unknown model name: {name}. Supported: logistic, cnn, slim_cnn, resnet18, resnet50, efficientnet_b0")
