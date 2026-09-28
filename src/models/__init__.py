"""
Models package for Limit Order Book prediction.
"""

from typing import Union
import torch.nn as nn

from .cnn_lob import LOB1DCNN
from .lstm_lob import LOBLSTM


def get_lob_model(
    model_type: str = "cnn",
    in_features: int = 43,
    lookback_window: int = 50,
    num_classes: int = 3,
    dropout: float = 0.2,
    hidden_dim: int = 64,
) -> nn.Module:
    """
    Factory function to instantiate LOB neural network model.
    """
    model_type = model_type.lower()
    if model_type == "cnn":
        return LOB1DCNN(
            in_features=in_features,
            lookback_window=lookback_window,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_type == "lstm":
        return LOBLSTM(
            in_features=in_features,
            lookback_window=lookback_window,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: 'cnn', 'lstm'")


__all__ = ["LOB1DCNN", "LOBLSTM", "get_lob_model"]
