import torch
import torch.nn as nn

from os import PathLike


class DeepConvNet(nn.Module):
    """
    Deep CNN for detecting table corner intersections.

    Outputs a probability [0,1] indicating if the center pixel is a corner.

    Example:
        >>> model = DeepConvNet(kernel_size=9, initial_filters=8, num_layers=7)
        >>> model.save("model.pth")
        >>>
        >>> # Later...
        >>> model = DeepConvNet.load("model.pth")
        >>> model.eval()
    """

    def __init__(
        self, kernel_size: int = 9, initial_filters: int = 8, num_layers: int = 7
    ):
        """
        Args:
            kernel_size: Convolution kernel size (affects receptive field)
            initial_filters: Starting number of filters (doubles each layer)
            num_layers: Network depth
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.initial_filters = initial_filters
        self.num_layers = num_layers

        # Build variable number of layers
        layers = []
        in_channels = 1
        out_channels = initial_filters

        for i in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=0,
                        bias=True,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
            out_channels = min(out_channels * 2, initial_filters * 8)  # Cap at 8x

        self.convs = nn.Sequential(*layers)
        self.conv_final = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.convs(x)
        x = self.conv_final(x)

        # Take center pixel
        center = x.shape[-1] // 2
        output = torch.sigmoid(x[:, :, center, center])

        return output.unsqueeze(1)

    def save(self, path: str | PathLike):
        """Save model with its configuration."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_config": {
                    "kernel_size": self.kernel_size,
                    "initial_filters": self.initial_filters,
                    "num_layers": self.num_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
