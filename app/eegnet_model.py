"""EEGNet definition shared by the notebook and the live demo app.

Kept byte-for-byte compatible with the architecture trained in
`notebooks/EA_FT_EEGNet_215526U (1).ipynb`, so exported state dicts load cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Compact CNN for EEG motor-imagery decoding (Lawhern et al., 2018)."""

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 22,
        n_samples: int = 1000,
        dropout_rate: float = 0.5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
    ) -> None:
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.block2(self.block1(dummy))
            self._flatten_size = out.numel()

        self.classifier = nn.Linear(self._flatten_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
