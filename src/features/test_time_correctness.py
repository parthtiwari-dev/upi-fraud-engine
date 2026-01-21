"""
Unit tests for Time Machine Logic.
Goal: Prove mathematically that features never leak future information.
"""

import unittest
import pandas as pd
from datetime import datetime

# Import your logic
from src.features.feature_definitions import compute_all_features
from src.features.schema import FeatureVector

class TestTimeCorrectness(unittest.TestCase):

    def setUp(self):
        """
        Prepare a synthetic micro-dataset for testing.
        """
        self.payer_id = "user_A"
        self.device_id = "device_X"

        # FIX 2: Default label time is in the future (no leakage by default)
        # This ensures the column is datetime64, not object/None
        default_label_time = pd.to_datetime("2024-01-01 10:30:00")

        # Create 3 transactions: T1, T2, T3 (Target), T4 (Future)
        self.data = [
            {
                "transaction_id": "t1",
                "event_timestamp": pd.to_datetime("2024-01-01 10:00:00"),
                "payer_id": self.payer_id,
                "device_id": self.device_id,
                "payee_id": "merchant_1",      # Added for completeness
                "payee_vpa": "merchant_1@upi", # FIX 1: Added payee_vpa
                "amount": 10.0,
                "is_fraud": 0,
                "label_available_timestamp": default_label_time
            },
            {
                "transaction_id": "t2",
                "event_timestamp": pd.to_datetime("2024-01-01 10:01:00"),
                "payer_id": self.payer_id,
                "device_id": self.device_id,
                "payee_id": "merchant_2",
                "payee_vpa": "merchant_2@upi",
                "amount": 20.0,
                "is_fraud": 0,
                "label_available_timestamp": default_label_time
            },
            {
                # TARGET TRANSACTION (We will compute features for this one)
                "transaction_id": "t3",
                "event_timestamp": pd.to_datetime("2024-01-01 10:02:00"),
                "payer_id": self.payer_id,
                "device_id": self.device_id,
                "payee_id": "merchant_3",
                "payee_vpa": "merchant_3@upi",
                "amount": 30.0,
                "is_fraud": 0,
                "label_available_timestamp": default_label_time
            },
            {
                # FUTURE TRANSACTION (Should never be seen)
                "transaction_id": "t4",
                "event_timestamp": pd.to_datetime("2024-01-01 10:10:00"),
                "payer_id": self.payer_id,
                "device_id": self.device_id,
                "payee_id": "merchant_4",
                "payee_vpa": "merchant_4@upi",
                "amount": 100.0,
                "is_fraud": 0,
                "label_available_timestamp": default_label_time
            }
        ]

        self.df = pd.DataFrame(self.data)

        # Double check types
        self.df["event_timestamp"] = pd.to_datetime(self.df["event_timestamp"])
        self.df["label_available_timestamp"] = pd.to_datetime(self.df["label_available_timestamp"])

        self.target_txn = self.data[2] # t3 at 10:02:00

    def test_velocity_spike(self):
        """
        Test Case 1: Velocity Calculation
        At 10:02 (t3), the 5-min window [09:57, 10:02) should include t1 and t2.
        It must strictly EXCLUDE t3 itself.
        """

        # Act
        fv: FeatureVector = compute_all_features(self.target_txn, self.df)

        # Assert
        # Expected: t1 ($10) + t2 ($20) = Count 2, Sum 30.0
        self.assertEqual(fv.payer_txn_count_5min, 2, "Count should be 2 (t1, t2)")
        self.assertEqual(fv.payer_txn_sum_5min, 30.0, "Sum should be 30.0")

        print("✅ test_velocity_spike Passed: Current transaction excluded correctly.")

    def test_future_leakage(self):
        """
        Test Case 2: Future Leakage
        At 10:02 (t3), we compute 1h velocity.
        It must NOT see t4 (10:10).
        """

        # Act
        fv: FeatureVector = compute_all_features(self.target_txn, self.df)

        # Assert
        # If t4 leaked, count would be 3 (t1, t2, t4 in 1h window)
        # Correct count is 2 (t1, t2)
        self.assertEqual(fv.payer_txn_count_1h, 2, "Future transaction t4 leaked into feature calculation!")

        print("✅ test_future_leakage Passed: Future transaction invisible.")

    def test_risk_history_label_delay(self):
        """
        Test Case 3: Label Leakage
        Scenario:
        - t1 was fraud.
        - But its label only arrives at 10:05.
        - We are at 10:02 (t3).
        - We should NOT know t1 is fraud yet.
        """

        # Modify t1 to be fraud
        self.df.loc[0, "is_fraud"] = 1

        # Scenario A: Label arrives LATE (10:05) -> Should NOT count
        self.df.loc[0, "label_available_timestamp"] = pd.to_datetime("2024-01-01 10:05:00")

        fv = compute_all_features(self.target_txn, self.df)
        self.assertEqual(fv.payer_past_fraud_count_30d, 0, "Leaked future label! (Scenario A)")

        # Scenario B: Label arrives EARLY (10:01) -> SHOULD count
        self.df.loc[0, "label_available_timestamp"] = pd.to_datetime("2024-01-01 10:01:00")

        fv_early = compute_all_features(self.target_txn, self.df)
        self.assertEqual(fv_early.payer_past_fraud_count_30d, 1, "Failed to count valid label! (Scenario B)")

        print("✅ test_risk_history_label_delay Passed: Delayed label respected.")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
