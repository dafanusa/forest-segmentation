import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPool2d(nn.Module):
    """
    2D SoftPool operation.

    Args:
        kernel_size: Ukuran kernel pooling.
        stride: Langkah pooling.
        padding: Padding.
        ceil_mode: Pengaturan ceil pada pooling.
    """

    def __init__(
        self,
        kernel_size=2,
        stride=None,
        padding=0,
        ceil_mode=False,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        exp_x = torch.exp(x)

        numerator = F.avg_pool2d(
            x * exp_x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )

        denominator = F.avg_pool2d(
            exp_x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )

        return numerator / (denominator + 1e-8)