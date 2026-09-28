"""
Long Short-Term Memory (LSTM) Neural Network for LOB Forecasting.
Processes sequential order book dynamics through recurrent gated cells.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LOBLSTM(nn.Module):
    """
    Recurrent LSTM architecture for LOB sequence modeling.
    Input tensor shape: (Batch_Size, Lookback_Window, Num_Features)
    Output tensor shape: (Batch_Size, 3) representing logits for {-1, 0, 1}.
    """

    def __init__(
        self,
        in_features: int = 43,
        lookback_window: int = 50,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super(LOBLSTM, self).__init__()
        self.in_features = in_features
        self.lookback_window = lookback_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Tensor of shape (Batch_Size, Lookback_Window, Num_Features)
        :return: Logits of shape (Batch_Size, 3)
        """
        # LSTM output: (Batch, Seq_Len, Hidden_Dim)
        out, _ = self.lstm(x)

        # Take the final temporal hidden state (most recent tick)
        last_hidden = out[:, -1, :]  # (Batch, Hidden_Dim)

        x = self.dropout(last_hidden)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        logits = self.fc_out(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Softmax probabilities across classes."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


if __name__ == "__main__":
    # Test required by Module 4 PRD:
    # "Passing a dummy tensor of shape (256, 50, 40) through the model outputs a tensor of shape (256, 3) without crashing."
    model = LOBLSTM(in_features=40, lookback_window=50, num_classes=3)
    dummy_input = torch.randn(256, 50, 40)
    output = model(dummy_input)
    probs = model.predict_proba(dummy_input)

    print(f"LSTM Logits output shape: {output.shape}")
    print(f"LSTM Probs output shape: {probs.shape}")
    assert output.shape == (256, 3), f"Unexpected shape {output.shape}"
    assert probs.shape == (256, 3), f"Unexpected shape {probs.shape}"
    print("Module 4 LSTM architecture test: PASSED successfully!")
