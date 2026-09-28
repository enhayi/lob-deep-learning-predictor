# Limit Order Book (LOB) Deep Learning Predictor

A modular, hardware-accelerated deep learning pipeline in PyTorch designed to forecast short-term mid-price movements from high-frequency Level-2 Limit Order Book (LOB) tick data.

Developed for local Linux/WSL2 compute environments accelerated by dedicated NVIDIA GPUs (CUDA 12.x/12.6, e.g. GeForce RTX 5060 Ti).

---

## 1. System Architecture & Modules

```
                    +------------------------------------+
                    |        LOB Data Sources            |
                    |  - Live Binance Depth Stream (100ms)|
                    |  - Realistic Synthetic Generator   |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 1: Ingestion & Storage     |
                    |  - 10 Asks + 10 Bids Parsing       |
                    |  - 10k-tick Buffer Auto-Flushing   |
                    |  - Snappy-compressed Parquet IO    |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 2: Feature Engineering     |
                    |  - Mid-Price & Bid-Ask Spread      |
                    |  - Order Book Imbalance (OBI)      |
                    |  - Causal Rolling Z-Score Norm     |
                    |  - Forward Smoothed Labels (-1,0,1)|
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 3: PyTorch Dataset/Loader  |
                    |  - Sliding Window (Lookback = 50)  |
                    |  - Non-leaking Chronological Split |
                    |  - Multi-worker Batching (256)     |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 4: Neural Architectures    |
                    |  - 1D Temporal CNN (DeepLOB-style) |
                    |  - Multi-layer Recurrent LSTM      |
                    |  - 3-Class Softmax Output          |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 5: Training & Acceleration |
                    |  - Dynamic CUDA Allocation         |
                    |  - Cross-Entropy Loss & Adam       |
                    |  - Gradient Clipping & Validation  |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Module 6: Backtesting & Strategy  |
                    |  - Precision, Recall, Macro F1     |
                    |  - Conviction-based Execution      |
                    |  - Transaction Fees (1 bp) & PnL   |
                    +------------------------------------+
```

---

## 2. Directory Structure

```
LOBDLP/
├── config/
│   └── config.yaml              # Global pipeline parameters & hyperparameters
├── src/
│   ├── ingestion/
│   │   ├── websocket_client.py  # Binance WebSocket partial depth client with 10k buffer
│   │   └── synthetic_data.py   # Realistic LOB snapshot generator and Parquet benchmark
│   ├── features/
│   │   └── feature_engineering.py # Microstructural features, rolling Z-score & labeling
│   ├── dataset/
│   │   └── lob_dataset.py       # Sliding window PyTorch Dataset & DataLoader
│   ├── models/
│   │   ├── cnn_lob.py           # 1D-CNN temporal convolution architecture
│   │   └── lstm_lob.py          # Recurrent LSTM architecture
│   ├── training/
│   │   └── trainer.py           # CUDA-enabled training loop with metric tracking
│   └── evaluation/
│       └── backtester.py        # Microstructural metrics & execution simulation
├── tests/
│   └── test_modules.py          # Pytest suite verifying PRD criteria for Modules 1-6
├── run_pipeline.py              # CLI orchestrator running the full pipeline end-to-end
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 3. Environment Setup (WSL2 / Linux)

### Conda Environment Creation
```bash
# Create and activate environment
conda create -n lob_env python=3.11 -y
conda activate lob_env

# Install PyTorch with CUDA 12.x acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install domain dependencies
pip install -r requirements.txt
```

---

## 4. Execution & Usage

### A. End-to-End Pipeline with Synthetic Data (Offline Verification)
Runs the entire pipeline through all 6 modules in seconds:
```bash
python run_pipeline.py --mode synthetic --num-ticks 15000 --epochs 5 --model-type cnn
```

### B. Live WebSocket Streaming (Binance Order Book)
Streams real-time Level-2 order book depth (10 bids and 10 asks updated every 100ms) directly into high-speed Parquet storage:
```bash
python run_pipeline.py --mode live --symbol btcusdt --num-ticks 10000 --epochs 5
```

Alternatively, run only the WebSocket client:
```bash
python src/ingestion/websocket_client.py --symbol btcusdt --buffer-size 10000
```

### C. Interactive Web Dashboard (Streamlit)
Launch the interactive web application locally:
```bash
streamlit run app.py
```

### D. Running Automated Module Tests
Execute the verification tests covering all 6 PRD milestones:
```bash
pytest tests/ -v
```

---


---

## 5. Module Highlights & Formulas

### Microstructural Features
* **Mid-Price**: $P_{mid, t} = \frac{P_{ask, 1, t} + P_{bid, 1, t}}{2}$
* **Bid-Ask Spread**: $S_t = P_{ask, 1, t} - P_{bid, 1, t}$
* **Order Book Imbalance (OBI)**:
  $$OBI_t = \frac{V_{bid, 1, t} - V_{ask, 1, t}}{V_{bid, 1, t} + V_{ask, 1, t}}$$
* **Causal Rolling Z-Score Normalization**:
  $$z_t = \frac{x_t - \mu_{W}}{\sigma_{W} + \epsilon}$$
  where statistics are computed strictly over past observations $(t - W : t)$ to eliminate lookahead bias.

### Categorical Target Labeling
* Future smoothed mid-price over horizon $k$ ticks:
  $$\bar{P}_{mid, future}(t) = \frac{1}{k} \sum_{j=1}^{k} P_{mid}(t + j)$$
* Return: $R_{future}(t) = \frac{\bar{P}_{mid, future}(t) - P_{mid}(t)}{P_{mid}(t)}$
* Classes:
  * $+1$ (Up): $R_{future}(t) > \theta$
  * $-1$ (Down): $R_{future}(t) < -\theta$
  * $0$ (Stationary): $|R_{future}(t)| \le \theta$

### Trading Strategy Simulation
* Position rule:
  * Long ($+1$) if $P(\text{Up}) > 0.70$
  * Short ($-1$) if $P(\text{Down}) > 0.70$
  * Flat ($0$) otherwise
* Transaction cost model:
  $$\text{Fee} = |\Delta \text{Position}| \times P_{mid} \times \frac{\text{fee\_bps}}{10000}$$
