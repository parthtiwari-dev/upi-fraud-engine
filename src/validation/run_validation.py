import pandas as pd
import duckdb
import great_expectations as gx


def validate_batch(df: pd.DataFrame) -> bool:
    """
    Gatekeeper for historical data.
    Uses the official GX 1.0 logic for runtime dataframes.
    """
    context = gx.get_context(context_root_dir="great_expectations")
    print("🔍 Running Batch Validation...")


    # 1. Connect the Data
    datasource_name = "police_datasource"
    # add_or_update_pandas works at the context level
    datasource = context.data_sources.add_or_update_pandas(name=datasource_name)
   
    # FIX: Use add_dataframe_asset. In GX 1.0, this is the idempotent call.
    # If it already exists, we catch it and get the existing one.
    try:
        asset = datasource.add_dataframe_asset(name="transactions_asset")
    except Exception:
        asset = datasource.get_asset("transactions_asset")
   
    # 2. Define how to slice the data
    # We use a try-except here as well to ensure idempotency
    try:
        batch_def = asset.add_batch_definition_whole_dataframe("batch_def")
    except Exception:
        batch_def = asset.get_batch_definition("batch_def")


    # 3. Retrieve the Suites
    try:
        schema_suite = context.suites.get("transaction_schema")
        business_suite = context.suites.get("business_logic")
    except (AttributeError, KeyError) as e:
        print(f"❌ Error: Suites not found. Run build_suite.py first. {e}")
        return False


    # 4. Create Validation Definitions (Linking Data + Suite)
    # These are the 'Police Rules' registered in the context
    val_schema = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="val_schema", data=batch_def, suite=schema_suite)
    )
    val_logic = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="val_logic", data=batch_def, suite=business_suite)
    )


    # 5. Create and Run Checkpoint
    checkpoint_name = "transaction_checkpoint"
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=checkpoint_name,
            validation_definitions=[val_schema, val_logic]
        )
    )


    # 6. Execute with the actual DataFrame
    # runtime_parameters provides the actual data to the batch definition
    result = checkpoint.run(batch_parameters={"dataframe": df})

    if not result.success:
        print("\n🚨 DATA VALIDATION FAILED!")
        
        # In GX 1.0, results are stored in run_results dict
        for run_id, run_result in result.run_results.items():
            # run_result is the validation result object itself
            if hasattr(run_result, 'results'):
                for r in run_result.results:
                    if not r.success:
                        # Print why it failed
                        print(f"❌ FAILED: {r.expectation_config.type}")
                        print(f"   Details: {r.result}")
        return False

    print("✅ Batch Validation Passed.")
    return True


def validate_streaming_event(event: dict) -> bool:
    """Lightweight streaming gate for the simulator."""
    return (
    event.get("amount", 0) > 0 and
    event.get("currency") == "INR" and
    event.get("label_available_timestamp") is None or
    event["label_available_timestamp"] > event["event_timestamp"]
    )



def main() -> None:
    db_path = "data/processed/transactions.duckdb"
    print(f"📦 Loading data from {db_path}...")


    try:
        con = duckdb.connect(db_path)
        df = con.execute("SELECT * FROM transactions ORDER BY event_timestamp ASC, transaction_id ASC").df()
        con.close()
        print(f"Rows: {len(df)} | Fraud Rate: {df['is_fraud'].mean():.4f}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ Database error: {e}")
        return


    if not validate_batch(df):
        raise RuntimeError("⛔ STOP! Data Validation Failed. Fix the data before training.")


    print("\n🎉 Phase 3 Step 2 Complete: Data is clean and safe for Phase 4.")


if __name__ == "__main__":
    main()
