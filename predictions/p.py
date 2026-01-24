import pandas as pd

file_path = r'c:\great learning self paced\z Final Projects\upi-fraud-engine\predictions\batch_predictions.csv'

import pandas as pd

import pandas as pd
import numpy as np

# Load submission
sub = pd.read_csv(file_path)

print("=" * 70)
print("SUBMISSION DIAGNOSTICS")
print("=" * 70)

# 1. Format check
print("\n1. FORMAT CHECK:")
print(f"   Columns: {sub.columns.tolist()}")
print(f"   Shape: {sub.shape}")
print(f"   Expected: ['transaction_id', 'isFraud']")

# 2. Prediction distribution
print("\n2. PREDICTION DISTRIBUTION:")
print(sub['isFraud'].describe())
print(f"   Mean: {sub['isFraud'].mean():.4f} (should be ~0.036)")
print(f"   % in [0.9, 1.0]: {(sub['isFraud'] > 0.9).mean():.2%}")
print(f"   % in [0.0, 0.1]: {(sub['isFraud'] < 0.1).mean():.2%}")

# 3. Inversion check
print("\n3. INVERSION CHECK:")
if sub['isFraud'].mean() > 0.5:
    print("   ❌ INVERTED! Mean > 0.5 suggests flipped predictions")
    print("   Fix: submission['isFraud'] = 1 - submission['isFraud']")
elif sub['isFraud'].mean() < 0.01:
    print("   ⚠️  Mean very low. Check if this matches training.")
else:
    print("   ✅ Mean looks reasonable")

# 4. Value range check
print("\n4. VALUE RANGE:")
print(f"   Min: {sub['isFraud'].min()}")
print(f"   Max: {sub['isFraud'].max()}")
if sub['isFraud'].min() < 0 or sub['isFraud'].max() > 1:
    print("   ❌ Values outside [0, 1] range!")

# 5. Sample predictions
print("\n5. SAMPLE PREDICTIONS:")
print(sub.head(10))

print("\n" + "=" * 70)
