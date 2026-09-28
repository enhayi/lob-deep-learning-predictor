"""
Feature Engineering and Categorical Labeling for Limit Order Book (LOB) Data.
Computes Mid-Price, Spread, Order Book Imbalance (OBI), relative prices,
applies causal rolling Z-score normalization, and calculates future smoothed mid-price labels.
"""

import logging
import sys
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LOBFeatureEngineering")


class LOBFeatureEngineer:
    """
    Transforms raw 10-level LOB snapshot DataFrames into model-ready engineered features
    and discrete forward-looking classification labels (-1, 0, 1).
    """

    def __init__(
        self,
        depth: int = 10,
        rolling_window: int = 100,
        future_horizon: int = 10,
        threshold: Optional[float] = 0.00005,
        use_quantile_threshold: bool = False,
    ):
        """
        :param depth: Number of bid and ask levels (default: 10)
        :param rolling_window: Window size for causal rolling Z-score normalization
        :param future_horizon: k ticks ahead for future smoothed mid-price calculation
        :param threshold: Percentage return threshold for labeling (None or float)
        :param use_quantile_threshold: If True, uses tertiles to guarantee balanced classes
        """
        self.depth = depth
        self.rolling_window = rolling_window
        self.future_horizon = future_horizon
        self.threshold = threshold
        self.use_quantile_threshold = use_quantile_threshold

    def compute_microstructural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute domain-specific microstructure signals:
        1. Mid-Price: (Best Ask + Best Bid) / 2
        2. Bid-Ask Spread: Best Ask - Best Bid
        3. Level 1 Order Book Imbalance: (BidVol1 - AskVol1) / (BidVol1 + AskVol1)
        4. Total Order Book Imbalance: (Sum(BidVol) - Sum(AskVol)) / (Sum(BidVol) + Sum(AskVol))
        5. Relative Price Levels: Price(level_i) - MidPrice
        """
        df = df.copy()

        # 1. Best Ask and Best Bid
        best_ask = df["ask_price_1"]
        best_bid = df["bid_price_1"]

        # 2. Mid-Price & Spread
        df["mid_price"] = (best_ask + best_bid) / 2.0
        df["spread"] = best_ask - best_bid

        # 3. Level 1 Order Book Imbalance (OBI)
        bid_vol_1 = df["bid_volume_1"]
        ask_vol_1 = df["ask_volume_1"]
        df["obi_level_1"] = (bid_vol_1 - ask_vol_1) / (bid_vol_1 + ask_vol_1 + 1e-8)

        # 4. Total Multi-Level Order Book Imbalance across all 10 levels
        total_bid_vol = sum(df[f"bid_volume_{i}"] for i in range(1, self.depth + 1))
        total_ask_vol = sum(df[f"ask_volume_{i}"] for i in range(1, self.depth + 1))
        df["obi_total"] = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol + 1e-8)

        # 5. Price offsets relative to mid-price (stationarity transform)
        for i in range(1, self.depth + 1):
            df[f"ask_price_diff_{i}"] = df[f"ask_price_{i}"] - df["mid_price"]
            df[f"bid_price_diff_{i}"] = df[f"bid_price_{i}"] - df["mid_price"]

        return df

    def apply_rolling_zscore_normalization(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply causal rolling Z-score normalization:
            z_t = (x_t - mean_w) / (std_w + epsilon)
        Prevents lookahead bias by computing statistics solely over historical ticks.
        """
        df = df.copy()
        eps = 1e-8

        for col in feature_columns:
            rolling = df[col].rolling(window=self.rolling_window, min_periods=self.rolling_window)
            r_mean = rolling.mean()
            r_std = rolling.std()

            norm_col_name = f"{col}_norm"
            df[norm_col_name] = (df[col] - r_mean) / (r_std + eps)

        return df

    def compute_categorical_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forward-looking mid-price movement categorical labels:
        - Smooth mid-price over future horizon k ticks:
            m_future(t) = mean(mid_price(t+1 : t+k))
        - Return: R(t) = (m_future(t) - mid_price(t)) / mid_price(t)
        - Labels:
            1  (Upward):    R(t) > threshold
            -1 (Downward):  R(t) < -threshold
            0  (Stationary): |R(t)| <= threshold
        - Mapped class index for PyTorch CrossEntropyLoss:
            0 -> Downward (-1)
            1 -> Stationary (0)
            2 -> Upward (1)
        """
        df = df.copy()
        k = self.future_horizon

        # Compute future smoothed mid-price using forward rolling mean
        # Using reversed rolling trick to look ahead k ticks:
        # m_future[t] = mean of mid_price[t+1 ... t+k]
        mid_prices = df["mid_price"].values
        n = len(mid_prices)
        future_smoothed = np.full(n, np.nan)

        # Fast vectorized forward window smoothing
        cumsum = np.cumsum(np.insert(mid_prices, 0, 0))
        for t in range(n - k):
            future_smoothed[t] = (cumsum[t + k + 1] - cumsum[t + 1]) / k

        df["future_smoothed_mid"] = future_smoothed
        df["future_return"] = (df["future_smoothed_mid"] - df["mid_price"]) / (df["mid_price"] + 1e-8)

        valid_returns = df["future_return"].dropna()

        # Determine threshold
        if self.use_quantile_threshold or (self.threshold is None):
            down_th = np.percentile(valid_returns, 33.3)
            up_th = np.percentile(valid_returns, 66.7)
            logger.info(f"Quantile-based thresholds: Down={down_th:.6f}, Up={up_th:.6f}")
        else:
            up_th = self.threshold
            down_th = -self.threshold
            logger.info(f"Fixed thresholds: Down={down_th:.6f}, Up={up_th:.6f}")

        # Assign labels: -1, 0, 1
        labels = np.zeros(n, dtype=np.int8)
        labels[df["future_return"] > up_th] = 1
        labels[df["future_return"] < down_th] = -1

        df["target_label"] = labels
        # PyTorch class indices: 0: Down (-1), 1: Stationary (0), 2: Up (1)
        df["target_class"] = df["target_label"] + 1

        return df

    def transform(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete end-to-end transformation pipeline.
        Returns cleaned DataFrame (rows with NaNs dropped) and list of normalized feature names.
        """
        logger.info(f"Starting feature engineering on {len(raw_df)} raw ticks...")

        # 1. Microstructural features
        df_micro = self.compute_microstructural_features(raw_df)

        # 2. Select raw features to normalize
        features_to_normalize = []
        for i in range(1, self.depth + 1):
            features_to_normalize.extend([
                f"ask_price_diff_{i}",
                f"ask_volume_{i}",
                f"bid_price_diff_{i}",
                f"bid_volume_{i}",
            ])
        features_to_normalize.extend(["spread", "obi_level_1", "obi_total"])

        # 3. Rolling Z-Score normalization
        df_norm = self.apply_rolling_zscore_normalization(df_micro, features_to_normalize)

        # 4. Forward categorical labeling
        df_labeled = self.compute_categorical_labels(df_norm)

        # Collect normalized feature column names
        norm_feature_cols = [f"{col}_norm" for col in features_to_normalize]

        # 5. Drop rows containing NaNs (warm-up rolling window & future lookahead tail)
        initial_count = len(df_labeled)
        df_clean = df_labeled.dropna(subset=norm_feature_cols + ["future_return"]).reset_index(drop=True)
        dropped_count = initial_count - len(df_clean)
        logger.info(
            f"Feature engineering complete. Kept {len(df_clean)} rows (dropped {dropped_count} warm-up/tail rows). "
            f"Total features: {len(norm_feature_cols)}"
        )

        # Log summary statistics
        self._log_feature_and_label_stats(df_clean, norm_feature_cols)

        return df_clean, norm_feature_cols

    def _log_feature_and_label_stats(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Log mean, std, and class balance statistics for verification."""
        means = df[feature_cols].mean()
        stds = df[feature_cols].std()
        logger.info(
            f"Normalized features stats across {len(feature_cols)} features: "
            f"Mean of Means = {means.mean():.4f}, Mean of Stds = {stds.mean():.4f}"
        )

        label_counts = df["target_label"].value_counts(normalize=True).to_dict()
        logger.info(f"Target label distribution: {label_counts}")


if __name__ == "__main__":
    import os
    from src.ingestion.synthetic_data import generate_synthetic_lob_dataset

    df_raw = generate_synthetic_lob_dataset(num_ticks=10000)
    fe = LOBFeatureEngineer(rolling_window=100, future_horizon=10, threshold=0.00005)
    df_clean, features = fe.transform(df_raw)
    print("Clean shape:", df_clean.shape)
    print("Features sample:", features[:5])
