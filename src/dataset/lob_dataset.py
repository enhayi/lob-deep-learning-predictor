"""
PyTorch Dataset and DataLoader for Limit Order Book (LOB) Sliding Window Sequences.
Transforms continuous tabular engineered LOB features into 3D tensors:
    (Batch_Size, Lookback_Window, Num_Features)
"""

import logging
import sys
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LOBDataset")


class LOBDataset(Dataset):
    """
    Sliding window dataset for high-frequency multivariate LOB time series.
    Yields sequences of shape (Lookback_Window, Num_Features) and corresponding target class.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback_window: int = 50,
    ):
        """
        :param features: 2D numpy array of shape (Total_Ticks, Num_Features)
        :param targets: 1D numpy array of shape (Total_Ticks,) with class labels (0, 1, 2)
        :param lookback_window: Sequence length N historical ticks (default: 50)
        """
        assert len(features) == len(targets), "Features and targets must have identical length"
        assert len(features) >= lookback_window, (
            f"Dataset length ({len(features)}) is shorter than lookback window ({lookback_window})"
        )

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.lookback_window = lookback_window
        self.num_samples = len(features) - lookback_window + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return sliding window sequence and target at the end of the window.
        - sequence: (lookback_window, num_features)
        - label: scalar class index
        """
        start_idx = idx
        end_idx = idx + self.lookback_window

        x = self.features[start_idx:end_idx]
        # Target corresponding to prediction point at end of lookback window
        y = self.targets[end_idx - 1]

        return x, y


def create_lob_dataloaders(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_class",
    lookback_window: int = 50,
    batch_size: int = 256,
    train_split: float = 0.8,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Chronologically splits data into training and validation sets,
    constructs LOBDatasets, and wraps them in PyTorch DataLoaders.

    :return: (train_loader, val_loader, num_features)
    """
    num_features = len(feature_cols)
    total_rows = len(df)
    split_idx = int(total_rows * train_split)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        f"Creating LOB DataLoaders: Total={total_rows} rows -> Train={len(train_df)}, Val={len(val_df)}"
    )

    # Extract NumPy arrays
    train_x = train_df[feature_cols].values.astype(np.float32)
    train_y = train_df[target_col].values.astype(np.int64)

    val_x = val_df[feature_cols].values.astype(np.float32)
    val_y = val_df[target_col].values.astype(np.int64)

    train_dataset = LOBDataset(train_x, train_y, lookback_window=lookback_window)
    val_dataset = LOBDataset(val_x, val_y, lookback_window=lookback_window)

    logger.info(
        f"Dataset sample counts: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, num_features


if __name__ == "__main__":
    # Quick self-test with dummy data
    dummy_x = np.random.randn(1000, 43).astype(np.float32)
    dummy_y = np.random.choice([0, 1, 2], size=1000).astype(np.int64)
    ds = LOBDataset(dummy_x, dummy_y, lookback_window=50)
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    batch_x, batch_y = next(iter(dl))
    print(f"Test batch x shape: {batch_x.shape}, batch y shape: {batch_y.shape}")
    assert batch_x.shape == (256, 50, 43), f"Unexpected shape {batch_x.shape}"
    print("Dataset verification PASSED!")
