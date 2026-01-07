import torch.nn as nn


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


def get_model(name: str, input_dim: int, num_classes: int):
    """Function for returning a model instance for the given name.

    """
    name = name.lower()
    if name in ('logistic', 'logistic_regression', 'linear', 'linear_model', 'linear_regression'):
        return LinearClassifier(input_dim, num_classes)
    raise ValueError(f"Unknown model name: {name}. Supported: logistic, linear")
    