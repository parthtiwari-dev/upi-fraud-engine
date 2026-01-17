# upi-fraud-engine
A high-frequency UPI fraud detection engine built with strict temporal discipline. Features a dual-path (batch/stream) architecture, stateful feature engineering in Redis, and a two-stage anomaly-supervised modeling pipeline.

## problem statement
At transaction time T, using only features computed from all transactions with timestamp < T,
decide: Alert or Not Alert.

## constraints
Constraint 1: We have a fixed alert budget of K alerts per day (e.g., 0.5% of daily volume).
Constraint 2: Label delay is ~48 hours; we train on labels available now but must decide on live data with no label.
Constraint 3: New device + small amount = hardest case (legitimate or sophisticated fraud?).

Success: Catch X% of fraud within budget while keeping false alert rate < Y%.
