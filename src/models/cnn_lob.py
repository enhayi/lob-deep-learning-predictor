"""
1D Convolutional Neural Network Architecture for Limit Order Book (LOB) Forecasting.
Processes multivariate time series snapshots of order book depth to extract spatial and temporal
microstructural patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LOB1DCNN(nn.Module):
    """
    1D-CNN architecture designed for multivariate LOB time-series classification.
    Input tensor shape: (Batch_Size, Lookback_Window, Num_Features)
    Output tensor shape: (Batch_Size, 3) representing logits for {-1, 0, 1}.
    """

    def __init__(
        self,
        in_features: int = 43,
        lookback_window: int = 50,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super(LOB1DCNN, self).__init__()
        self.in_features = in_features
        self.lookback_window = lookback_window
        self.num_classes = num_classes

        # Temporal Convolutions along sequence dimension
        # Input to Conv1d: (Batch, in_features, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)

        # Classification head
        self.fc1 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Tensor of shape (Batch_Size, Lookback_Window, Num_Features)
        :return: Logits of shape (Batch_Size, 3)
        """
        # Permute (Batch, Seq_Len, Features) -> (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)

        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dropout(x)

        # Block 2
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.pool(x)
        x = self.dropout(x)

        # Block 3
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = self.adaptive_pool(x)  # (Batch, 64, 1)

        # Flatten
        x = x.squeeze(-1)  # (Batch, 64)

        # Dense classification head
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (Batch, 3)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Softmax probabilities across classes:
        Class 0: Downward (-1)
        Class 1: Stationary (0)
        Class 2: Upward (1)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


if __name__ == "__main__":
    # Test required by Module 4 PRD:
    # "Passing a dummy tensor of shape (256, 50, 40) through the model outputs a tensor of shape (256, 3) without crashing."
    model = LOB1DCNN(in_features=40, lookback_window=50, num_classes=3)
    dummy_input = torch.randn(256, 50, 40)
    output = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    print(f"Logits output shape: {output.shape}")
    print(f"Probs output shape: {probs.shape}")
    assert output.shape == (256, 3), f"Unexpected shape {output.shape}"
    assert probs.shape == (256, 3), f"Unexpected shape {probs.shape}"
    print("Module 4 CNN architecture test: PASSED successfully!")
