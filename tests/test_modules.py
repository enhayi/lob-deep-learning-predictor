"""
Unit and Integration Tests for LOB Deep Learning Predictor.
Covers PRD Acceptance Criteria for Modules 1 through 6.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

from src.ingestion.synthetic_data import (
    generate_synthetic_lob_dataset,
    save_lob_to_parquet,
    verify_parquet_load_speed,
)
from src.features.feature_engineering import LOBFeatureEngineer
from src.dataset.lob_dataset import LOBDataset, create_lob_dataloaders
from src.models import get_lob_model
from src.models.cnn_lob import LOB1DCNN
from src.models.lstm_lob import LOBLSTM
from src.training.trainer import LOBTrainer
from src.evaluation.backtester import LOBBacktester


@pytest.fixture(scope="session")
def sample_lob_df():
    """Generates synthetic LOB dataset for testing."""
    return generate_synthetic_lob_dataset(num_ticks=1200, seed=42)


def test_module_1_parquet_io(sample_lob_df):
    """
    Module 1 Test: Script successfully outputs a data.parquet file
    that can be loaded in under 2 seconds.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path = os.path.join(tmp_dir, "test_lob.parquet")
        save_lob_to_parquet(sample_lob_df, parquet_path)
        assert os.path.exists(parquet_path), "Parquet file was not written"

        passed, load_time, rows = verify_parquet_load_speed(parquet_path, max_expected_seconds=2.0)
        assert passed, f"Parquet loading took {load_time:.4f}s, exceeding 2.0s!"
        assert rows == len(sample_lob_df), "Loaded rows do not match original"


def test_module_2_feature_engineering(sample_lob_df):
    """
    Module 2 Test: Output statistics show normalized features with mu ~ 0 and sigma ~ 1,
    and valid discrete labels (-1, 0, 1).
    """
    fe = LOBFeatureEngineer(rolling_window=50, future_horizon=5, threshold=0.00005)
    df_clean, feature_cols = fe.transform(sample_lob_df)

    assert len(df_clean) > 0, "No clean rows generated"
    assert len(feature_cols) > 0, "No features generated"

    # Verify normalization statistics (mean ~ 0, std ~ 1)
    means = df_clean[feature_cols].mean()
    stds = df_clean[feature_cols].std()

    assert np.all(np.abs(means) < 0.5), f"Means not centered: {means.mean()}"
    assert np.all(stds > 0.5) and np.all(stds < 1.5), f"Stds not near 1: {stds.mean()}"

    # Verify labels contain valid classes
    unique_labels = set(df_clean["target_label"].unique())
    assert unique_labels.issubset({-1, 0, 1}), f"Invalid labels found: {unique_labels}"
    unique_classes = set(df_clean["target_class"].unique())
    assert unique_classes.issubset({0, 1, 2}), f"Invalid target classes: {unique_classes}"


def test_module_3_dataset_dataloader(sample_lob_df):
    """
    Module 3 Test: Iterating over DataLoader yields a batch of shape
    (Batch_Size, Lookback_Window, Num_Features).
    """
    fe = LOBFeatureEngineer(rolling_window=50, future_horizon=5)
    df_clean, feature_cols = fe.transform(sample_lob_df)

    lookback_window = 30
    batch_size = 64
    train_loader, val_loader, num_features = create_lob_dataloaders(
        df_clean,
        feature_cols,
        lookback_window=lookback_window,
        batch_size=batch_size,
        num_workers=0,
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (batch_size, lookback_window, num_features), (
        f"Unexpected batch_x shape {batch_x.shape}"
    )
    assert batch_y.shape == (batch_size,), f"Unexpected batch_y shape {batch_y.shape}"


def test_module_4_model_architectures():
    """
    Module 4 Test: Passing dummy tensor of shape (256, 50, 40)
    through the model outputs a tensor of shape (256, 3) without crashing.
    """
    batch_size = 256
    seq_len = 50
    num_feats = 40
    dummy_input = torch.randn(batch_size, seq_len, num_feats)

    # 1. Test CNN
    cnn_model = LOB1DCNN(in_features=num_feats, lookback_window=seq_len, num_classes=3)
    cnn_out = cnn_model(dummy_input)
    cnn_probs = cnn_model.predict_proba(dummy_input)
    assert cnn_out.shape == (batch_size, 3), f"CNN output shape mismatch: {cnn_out.shape}"
    assert cnn_probs.shape == (batch_size, 3), f"CNN probs shape mismatch: {cnn_probs.shape}"

    # 2. Test LSTM
    lstm_model = LOBLSTM(in_features=num_feats, lookback_window=seq_len, num_classes=3)
    lstm_out = lstm_model(dummy_input)
    lstm_probs = lstm_model.predict_proba(dummy_input)
    assert lstm_out.shape == (batch_size, 3), f"LSTM output shape mismatch: {lstm_out.shape}"
    assert lstm_probs.shape == (batch_size, 3), f"LSTM probs shape mismatch: {lstm_probs.shape}"


def test_module_5_training_execution(sample_lob_df):
    """
    Module 5 Test: Model successfully executes training loop and demonstrates
    loss computation and optimizer stepping.
    """
    fe = LOBFeatureEngineer(rolling_window=50, future_horizon=5)
    df_clean, feature_cols = fe.transform(sample_lob_df)

    train_loader, val_loader, num_features = create_lob_dataloaders(
        df_clean,
        feature_cols,
        lookback_window=20,
        batch_size=32,
        num_workers=0,
    )

    model = get_lob_model("cnn", in_features=num_features, lookback_window=20)
    trainer = LOBTrainer(model=model, learning_rate=0.005)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer.checkpoint_dir = tmp_dir
        history = trainer.fit(train_loader, val_loader, epochs=2, save_best=False)

    assert len(history["train_loss"]) == 2, "Did not complete 2 epochs"
    assert history["train_loss"][0] > 0, "Initial loss should be positive"


def test_module_6_backtest_and_metrics():
    """
    Module 6 Test: Metrics calculation (Precision, Recall, F1)
    and strategy simulation with transaction fee deduction.
    """
    n = 200
    y_true = np.random.choice([0, 1, 2], size=n)
    y_pred = np.random.choice([0, 1, 2], size=n)

    backtester = LOBBacktester(confidence_threshold=0.6, transaction_fee_bps=1.0)
    metrics = backtester.compute_metrics(y_true, y_pred)

    assert "classification_report" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    dummy_probs = np.random.dirichlet((1, 1, 1), size=n)
    dummy_prices = 60000.0 + np.cumsum(np.random.randn(n) * 5.0)

    sim = backtester.simulate_strategy(dummy_probs, dummy_prices)
    assert "total_net_pnl" in sim
    assert "total_fees" in sim
    assert sim["total_fees"] >= 0.0
