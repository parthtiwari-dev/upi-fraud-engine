"""Goal: Combine train and test into one big timeline.
What to Do:
1. Create a file: data_generation/generate_data.py.
2. Write the main logic (no code, just steps):
Step A: Process Train
• Load train_transaction.csv + train_identity.csv.
• Apply standardize_columns(df, is_train_set=True).
• Apply map_to_upi(df).
• You now have a dataframe with labeled fraud (is_fraud is 0 or 1).
Step B: Process Test
• Load test_transaction.csv + test_identity.csv.
• Apply standardize_columns(df, is_train_set=False).
• Apply map_to_upi(df).
• You now have a dataframe where is_fraud is None (no labels).
Step C: Combine
• Stack train and test: full_df = concat([train, test]).
• Sort by time: full_df = full_df.sort_values('event_timestamp').
• Reset the index.
Stop Condition:
You have one dataframe where:
• Early rows (train) have labels.
• Later rows (test) have no labels.
Everything is sorted by time.
"""


import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent)) 
from data_generation.enrich_to_upi_schema import load_and_merge_identity, standardize_columns, map_to_upi


def generate_full_dataset():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
   
    print("Starting Data Generation Pipeline.")

    df_train = load_and_merge_identity(
        raw_dir / 'train_transaction.csv',
        raw_dir / "train_identity.csv")
    
    df_train = standardize_columns(df_train, is_train_set=True)
    df_train = map_to_upi(df_train)

    print(f"shape of train dataset is {df_train.shape}")

    # processing test 

    df_test = load_and_merge_identity(
        raw_dir / "test_transaction.csv",
        raw_dir / "test_identity.csv"
    )

    df_test = standardize_columns(df_test, is_train_set=False)
    df_test = map_to_upi(df_test)

    print(f"shape of test dataset is {df_test.shape}")

    print("\n Combining and Sorting...")
    full_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    del df_test,df_train
    print(f"shape of full dataset is {full_df.shape}")
    print("Sorting by event_timestamp...")
    full_df.sort_values(by="event_timestamp", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    output_path = processed_dir / "full_upi_dataset.csv"
    full_df.to_csv(output_path, index=False)
    print(" DONE! Unified dataset created.")

if __name__ == "__main__":
    generate_full_dataset()

