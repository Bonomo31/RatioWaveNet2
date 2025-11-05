"""Temporal Convolutional Network ensemble for fusing multiple EEG streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Basic TCN residual block with two dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        return out + residual


@dataclass
class TCNEnsembleConfig:
    """Configuration used to build :class:`TCNEnsemble`."""

    embed_dim: int = 16
    hidden_dim: int = 96
    levels: int = 5
    kernel_size: int = 5
    dropout: float = 0.1


class TCNEnsemble(nn.Module):
    """Fuse RatioWaveNet, TCFormer and optional raw EEG streams with a TCN."""

    def __init__(
        self,
        rwn_channels: int,
        tcf_channels: int,
        *,
        signal_channels: Optional[int],
        n_classes: int,
        config: TCNEnsembleConfig,
    ) -> None:
        super().__init__()
        self.has_signal = signal_channels is not None and signal_channels > 0

        self.rwn_in = nn.Conv1d(rwn_channels, config.embed_dim, kernel_size=1)
        self.tcf_in = nn.Conv1d(tcf_channels, config.embed_dim, kernel_size=1)
        if self.has_signal:
            assert signal_channels is not None
            self.sig_in = nn.Conv1d(signal_channels, config.embed_dim, kernel_size=1)

        total_channels = config.embed_dim * (3 if self.has_signal else 2)
        blocks = []
        in_ch = total_channels
        for level in range(config.levels):
            blocks.append(
                TemporalConvBlock(
                    in_ch,
                    config.hidden_dim,
                    kernel_size=config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                )
            )
            in_ch = config.hidden_dim

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(config.hidden_dim, n_classes)

    def forward(
        self,
        rwn: torch.Tensor,
        tcf: torch.Tensor,
        signal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        streams = [self.rwn_in(rwn), self.tcf_in(tcf)]
        if self.has_signal and signal is not None:
            streams.append(self.sig_in(signal))
        x = torch.cat(streams, dim=1)
        x = self.tcn(x)
        x = x.mean(dim=-1)
        return self.head(x)


__all__ = ["TCNEnsemble", "TCNEnsembleConfig", "TemporalConvBlock"]
