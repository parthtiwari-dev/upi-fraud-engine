import great_expectations as gx

def main():
    context = gx.get_context(context_root_dir="great_expectations")

    datasource_name = "pandas_datasource"

    # Idempotent rebuild
    try:
        context.data_sources.delete_pandas(datasource_name)
    except Exception:
        pass

    # Register Pandas datasource (Fluent API)
    datasource = context.data_sources.add_pandas(
        name=datasource_name
    )

    # Add runtime dataframe asset
    datasource.add_dataframe_asset(name="transactions_df")

    print("✅ Pandas Fluent datasource registered.")

if __name__ == "__main__":
    main()
