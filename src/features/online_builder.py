"""
Streaming Feature Engine (Online Store).
Maintains in-memory state to produce bit-identical features to the offline builder.
"""

import pandas as pd
from typing import Dict, Deque
from collections import deque, defaultdict
from datetime import timedelta

from src.features.feature_definitions import compute_all_features
from src.features.schema import FeatureVector


class OnlineFeatureStore:

    def __init__(self):
        self.payer_history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=1000))
        self.device_history: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=1000))
        self.MAX_WINDOW = timedelta(days=30)  # Still used for risk history

    def get_features(self, transaction: dict) -> FeatureVector:
        current_time = pd.to_datetime(transaction["event_timestamp"])
        payer_id = transaction["payer_id"]
        device_id = transaction["device_id"]

        payer_txns = list(self.payer_history[payer_id])
        device_txns = list(self.device_history[device_id])

        history = {t["transaction_id"]: t for t in payer_txns + device_txns}
        history_df = pd.DataFrame(history.values())

        if history_df.empty:
            history_df = pd.DataFrame(columns=[
                "transaction_id", "event_timestamp", "amount",
                "payer_id", "device_id", "payee_vpa",
                "is_fraud", "label_available_timestamp"
            ])
        else:
            history_df["event_timestamp"] = pd.to_datetime(history_df["event_timestamp"])
            if "label_available_timestamp" in history_df.columns:
                history_df["label_available_timestamp"] = pd.to_datetime(history_df["label_available_timestamp"])

        return compute_all_features(transaction, history_df)

    def ingest(self, transaction: dict):
        payer_id = transaction["payer_id"]
        device_id = transaction["device_id"]
        current_time = pd.to_datetime(transaction["event_timestamp"])

        if self.payer_history[payer_id]:
            last_time = pd.to_datetime(self.payer_history[payer_id][-1]["event_timestamp"])
            assert last_time <= current_time, "Out-of-order event detected"

        self.payer_history[payer_id].append(transaction)
        self.device_history[device_id].append(transaction)

        cutoff = current_time - self.MAX_WINDOW
                    
        # Deques with maxlen auto-prune, but still prune old events for risk history
        # (Only needed for entities beyond 1000 rows)
        def prune(dq: Deque[dict]):
            while len(dq) > 1000:  # Extra safety (maxlen should handle this)
                if pd.to_datetime(dq[0]["event_timestamp"]) < cutoff:
                    dq.popleft()
                else:
                    break

        # Note: payer/device deques auto-prune to 1000 via maxlen
        # This prune is mainly for time-based risk history cleanup

        prune(self.payer_history[payer_id])
        prune(self.device_history[device_id])
