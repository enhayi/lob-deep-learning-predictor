"""
WebSocket Client for High-Frequency Limit Order Book (LOB) Ingestion.
Subscribes to exchange partial book depth streams (e.g., Binance depth10@100ms),
parses JSON payloads into continuous tabular records, and flushes to Parquet files
in batches of 10,000 rows to ensure zero memory leakage.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LOBWebSocketClient")


class LOBWebSocketClient:
    """
    Subscribes to Binance Partial Book Depth Streams (Top 10 Bids and Top 10 Asks).
    Handles real-time JSON parsing, buffer management, memory-efficient Parquet flushes,
    and automatic reconnection on network drops.
    """

    def __init__(
        self,
        symbol: str = "btcusdt",
        depth: int = 10,
        update_speed: str = "100ms",
        buffer_size: int = 10000,
        output_dir: str = "data",
        output_filename: str = "raw_lob_data.parquet",
    ):
        """
        Initialize the WebSocket Client.

        :param symbol: Ticker symbol (lowercase, e.g., 'btcusdt')
        :param depth: Order book depth levels (e.g., 10 or 20)
        :param update_speed: Stream update interval (e.g., '100ms' or '1000ms')
        :param buffer_size: Number of records before auto-flushing to disk
        :param output_dir: Local directory for Parquet files
        :param output_filename: Target output file name
        """
        self.symbol = symbol.lower()
        self.depth = depth
        self.update_speed = update_speed
        self.buffer_size = buffer_size
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.output_filepath = os.path.join(output_dir, output_filename)

        self.stream_url = (
            f"wss://stream.binance.com:9443/ws/{self.symbol}@depth{self.depth}@{self.update_speed}"
        )

        # Internal state buffer
        self.buffer: List[Dict[str, Any]] = []
        self.total_recorded_ticks = 0
        self.flush_count = 0
        self.running = False
        self._schema = self._create_arrow_schema()

        os.makedirs(self.output_dir, exist_ok=True)

    def _create_arrow_schema(self) -> pa.Schema:
        """Create explicit PyArrow Schema for consistent, type-safe Parquet files."""
        fields = [("timestamp", pa.int64())]
        for i in range(1, self.depth + 1):
            fields.append((f"ask_price_{i}", pa.float64()))
            fields.append((f"ask_volume_{i}", pa.float64()))
        for i in range(1, self.depth + 1):
            fields.append((f"bid_price_{i}", pa.float64()))
            fields.append((f"bid_volume_{i}", pa.float64()))
        return pa.schema(fields)

    def parse_payload(self, raw_message: str) -> Optional[Dict[str, Any]]:
        """
        Parse raw incoming JSON payload into flattened LOB snapshot dictionary.

        Expected schema from Binance @depth10:
        {
          "lastUpdateId": 1234567,
          "bids": [["price_str", "qty_str"], ...],
          "asks": [["price_str", "qty_str"], ...]
        }
        """
        try:
            payload = json.loads(raw_message)
            bids = payload.get("bids", [])
            asks = payload.get("asks", [])

            # High-resolution UNIX timestamp in milliseconds
            # If payload includes event time 'E', use it; otherwise, use system time (ms)
            timestamp = payload.get("E", int(time.time() * 1000))

            if len(bids) < self.depth or len(asks) < self.depth:
                logger.warning(
                    f"Incomplete depth received: bids={len(bids)}, asks={len(asks)}. Expected={self.depth}"
                )
                return None

            record: Dict[str, Any] = {"timestamp": timestamp}

            # Top 10 Asks (lowest ask is best ask: index 0)
            for i in range(self.depth):
                level = i + 1
                record[f"ask_price_{level}"] = float(asks[i][0])
                record[f"ask_volume_{level}"] = float(asks[i][1])

            # Top 10 Bids (highest bid is best bid: index 0)
            for i in range(self.depth):
                level = i + 1
                record[f"bid_price_{level}"] = float(bids[i][0])
                record[f"bid_volume_{level}"] = float(bids[i][1])

            return record

        except Exception as e:
            logger.error(f"Error parsing message payload: {e}")
            return None

    def flush_buffer(self) -> None:
        """
        Flush in-memory buffer to Parquet storage on local disk,
        then clear the buffer to prevent memory leaks.
        """
        if not self.buffer:
            return

        start_time = time.time()
        num_rows = len(self.buffer)
        df = pd.DataFrame(self.buffer)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)

        # Append to existing Parquet dataset or write new file
        if not os.path.exists(self.output_filepath):
            pq.write_table(table, self.output_filepath, compression="snappy")
        else:
            # Read existing table and concatenate, or use multi-chunk file
            existing_table = pq.read_table(self.output_filepath)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, self.output_filepath, compression="snappy")

        elapsed = time.time() - start_time
        self.flush_count += 1
        self.total_recorded_ticks += num_rows

        logger.info(
            f"Flushed {num_rows} rows to '{self.output_filepath}' in {elapsed:.3f}s. "
            f"Total ticks saved: {self.total_recorded_ticks} (Flush #{self.flush_count})"
        )

        # Free memory immediately
        self.buffer.clear()
        del df
        del table

    async def connect_and_stream(self, max_ticks: Optional[int] = None) -> None:
        """
        Manage WebSocket connection with automatic retry and exponential backoff.

        :param max_ticks: Optional limit on total ticks to capture before terminating (useful for tests).
        """
        self.running = True
        backoff_seconds = 1.0
        max_backoff = 30.0

        logger.info(f"Connecting to WebSocket stream: {self.stream_url}")

        while self.running:
            try:
                async with websockets.connect(
                    self.stream_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("Connected successfully to exchange stream.")
                    backoff_seconds = 1.0  # Reset backoff upon successful connection

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            record = self.parse_payload(msg)

                            if record is not None:
                                self.buffer.append(record)

                                if len(self.buffer) >= self.buffer_size:
                                    self.flush_buffer()

                                if max_ticks and (self.total_recorded_ticks + len(self.buffer)) >= max_ticks:
                                    logger.info(f"Reached target tick count: {max_ticks}. Halting stream.")
                                    self.running = False
                                    break

                        except asyncio.TimeoutError:
                            # Send ping to verify connection liveness
                            pong_waiter = await ws.ping()
                            await asyncio.wait_for(pong_waiter, timeout=5.0)

            except (websockets.ConnectionClosed, websockets.WebSocketException, OSError) as e:
                logger.warning(f"Connection dropped: {e}. Reconnecting in {backoff_seconds:.1f}s...")
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)

            except asyncio.CancelledError:
                logger.info("WebSocket listener task cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in streaming loop: {e}", exc_info=True)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, max_backoff)

        # Ensure any remaining data in the buffer is flushed before exiting
        if self.buffer:
            logger.info("Flushing final remaining records before shutdown...")
            self.flush_buffer()

    def stop(self) -> None:
        """Gracefully stop the streaming loop."""
        logger.info("Stopping WebSocket client...")
        self.running = False


def run_streamer(
    symbol: str = "btcusdt",
    buffer_size: int = 10000,
    max_ticks: Optional[int] = None,
    output_dir: str = "data",
    output_filename: str = "raw_lob_data.parquet",
):
    """Entry point for executing the streaming ingestion service."""
    client = LOBWebSocketClient(
        symbol=symbol,
        buffer_size=buffer_size,
        output_dir=output_dir,
        output_filename=output_filename,
    )

    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, client.stop)
        except NotImplementedError:
            # Signal handlers not implemented on Windows event loop
            pass

    try:
        loop.run_until_complete(client.connect_and_stream(max_ticks=max_ticks))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        client.stop()
        if client.buffer:
            client.flush_buffer()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOB WebSocket Stream Ingestion Client")
    parser.add_argument("--symbol", type=str, default="btcusdt", help="Trading pair symbol (default: btcusdt)")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Buffer size before flushing to parquet")
    parser.add_argument("--max-ticks", type=int, default=None, help="Max ticks to collect (optional)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--output-filename", type=str, default="raw_lob_data.parquet", help="Parquet file name")

    args = parser.parse_args()
    run_streamer(
        symbol=args.symbol,
        buffer_size=args.buffer_size,
        max_ticks=args.max_ticks,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
