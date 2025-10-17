import torch
import torch.nn as nn

from os import PathLike


class DeepConvNet(nn.Module):
    def __init__(
        self, kernel_size: int = 9, initial_filters: int = 8, num_layers: int = 7
    ):
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
    def load(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
