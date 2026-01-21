import pandas as pd
from src.features.offline_builder import build_features_batch
from src.features.online_builder import OnlineFeatureStore

def test_offline_online_parity():
    """
    Proves that offline (truth engine) and online (streaming engine)
    produce identical feature vectors for every transaction.
    """

    # Load small deterministic slice
    df = pd.read_parquet("data/processed/parity_sample.parquet")

    # 1. Offline Truth
    offline_df = build_features_batch(df)

    # 2. Online Streaming
    store = OnlineFeatureStore()
    online_rows = []

    for _, row in df.iterrows():
        txn = row.to_dict()
        fv = store.get_features(txn)
        store.ingest(txn)
        online_rows.append(fv.model_dump())

    online_df = pd.DataFrame(online_rows)

    # 3. Compare
    pd.testing.assert_frame_equal(
        offline_df.sort_index(axis=1),
        online_df.sort_index(axis=1),
        check_dtype=False
    )
