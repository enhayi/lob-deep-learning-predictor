"""
Backtesting and Evaluation Module for Limit Order Book Predictor.
Computes multi-class classification metrics (Precision, Recall, F1)
and simulates an execution strategy with transaction fee deduction to calculate net PnL.
"""

import logging
import sys
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LOBBacktester")


class LOBBacktester:
    """
    Evaluates predictive accuracy through microstructural classification metrics
    and simulates quantitative strategy execution.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        transaction_fee_bps: float = 1.0,
    ):
        """
        :param confidence_threshold: Probability threshold for taking high-conviction trades (default: 0.7)
        :param transaction_fee_bps: Trading fee in basis points (1 bp = 0.01% = 0.0001)
        """
        self.confidence_threshold = confidence_threshold
        self.transaction_fee_bps = transaction_fee_bps
        self.fee_rate = transaction_fee_bps / 10000.0

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute multi-class Precision, Recall, and F1-Scores.
        Classes mapped: 0 -> Down (-1), 1 -> Stationary (0), 2 -> Up (1).
        """
        target_names = ["Down (-1)", "Stationary (0)", "Up (1)"]

        # Classification report dictionary
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        logger.info("\n" + classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=target_names,
            zero_division=0,
        ))

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }

    def simulate_strategy(
        self,
        probabilities: np.ndarray,
        mid_prices: np.ndarray,
        future_horizon: int = 10,
    ) -> Dict[str, Any]:
        """
        Simulate quantitative trading strategy:
        - Buy (Long: +1) if P(Up) > threshold
        - Sell (Short: -1) if P(Down) > threshold
        - Flat (0) otherwise
        Calculates Gross PnL, Fees, Net PnL, Sharpe, and Drawdowns.

        :param probabilities: Array of shape (N, 3) where cols are [P(Down), P(Stationary), P(Up)]
        :param mid_prices: Array of shape (N,) corresponding to mid-price at each tick
        :param future_horizon: Holding horizon or tick interval
        """
        n = len(probabilities)
        assert n == len(mid_prices), "Probabilities and prices must have identical length"

        prob_down = probabilities[:, 0]
        prob_up = probabilities[:, 2]

        positions = np.zeros(n, dtype=np.int8)

        # Strategy rule: Buy when P(Up) > threshold, Short when P(Down) > threshold
        long_signals = prob_up > self.confidence_threshold
        short_signals = prob_down > self.confidence_threshold

        positions[long_signals] = 1
        positions[short_signals] = -1

        # Price returns over each tick step: (P_{t} - P_{t-1})
        price_diffs = np.zeros(n)
        price_diffs[1:] = np.diff(mid_prices)

        # Position held during period t is the position decided at t-1
        held_positions = np.zeros(n)
        held_positions[1:] = positions[:-1]

        gross_pnl = held_positions * price_diffs

        # Position changes (trades executed)
        position_changes = np.zeros(n)
        position_changes[0] = abs(positions[0])
        position_changes[1:] = np.abs(np.diff(positions))

        # Transaction fees in quote currency: |position_change| * mid_price * fee_rate
        fees = position_changes * mid_prices * self.fee_rate

        net_pnl = gross_pnl - fees
        cumulative_gross_pnl = np.cumsum(gross_pnl)
        cumulative_net_pnl = np.cumsum(net_pnl)

        # Peak and Maximum Drawdown of cumulative net PnL
        peak = np.maximum.accumulate(cumulative_net_pnl)
        drawdown = peak - cumulative_net_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Trade statistics
        total_trades = int(np.sum(position_changes > 0))
        trade_indices = np.where(held_positions != 0)[0]
        active_returns = net_pnl[trade_indices] if len(trade_indices) > 0 else np.array([0.0])

        win_rate = (
            float(np.mean(active_returns > 0)) if len(active_returns) > 0 else 0.0
        )

        # Annualized / Tick Sharpe Ratio
        if np.std(net_pnl) > 1e-9:
            sharpe_ratio = float(np.mean(net_pnl) / np.std(net_pnl) * np.sqrt(10000))
        else:
            sharpe_ratio = 0.0

        results = {
            "total_samples": n,
            "total_trades": total_trades,
            "total_gross_pnl": float(cumulative_gross_pnl[-1]),
            "total_fees": float(np.sum(fees)),
            "total_net_pnl": float(cumulative_net_pnl[-1]),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe_ratio),
            "long_signals_count": int(np.sum(long_signals)),
            "short_signals_count": int(np.sum(short_signals)),
        }

        logger.info(
            f"Strategy Backtest Results:\n"
            f"  Trades: {results['total_trades']} | Longs: {results['long_signals_count']} | Shorts: {results['short_signals_count']}\n"
            f"  Gross PnL: ${results['total_gross_pnl']:.2f}\n"
            f"  Total Fees (1bp): ${results['total_fees']:.2f}\n"
            f"  Net PnL: ${results['total_net_pnl']:.2f}\n"
            f"  Win Rate: {results['win_rate'] * 100:.2f}%\n"
            f"  Max Drawdown: ${results['max_drawdown']:.2f}\n"
            f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}"
        )

        return results


if __name__ == "__main__":
    # Test simulation
    n = 1000
    dummy_probs = np.random.dirichlet((1, 1, 1), size=n)
    dummy_prices = 65000.0 + np.cumsum(np.random.randn(n) * 2.0)

    backtester = LOBBacktester(confidence_threshold=0.7, transaction_fee_bps=1.0)
    res = backtester.simulate_strategy(dummy_probs, dummy_prices)
    print("Simulation test passed!")
