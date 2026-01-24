import great_expectations as gx
import great_expectations.expectations as gxe

def build_schema_suite(context):
    suite_name = "transaction_schema"
    
    # CORRECT WAY: Ask the Context to give you a suite (create or get existing)
    # This ensures the suite is properly registered, versioned, and owned by the context immediately.
    # In GX 1.0, 'suites.add' is the registry method, but we must construct the object 
    # effectively as a definition that the context *adopts*.
    
    # 1. Clean slate (optional but good for idempotent build scripts)
    try:
        context.suites.delete(suite_name)
    except Exception:
        pass

    # 2. Create the Suite *through* the context's registry mechanism
    # We instantiate it, but immediately register it to get the "Context-Aware" object back.
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # 3. Add Expectations to the *Context-Owned* Suite
    # (The suite object returned by context.suites.add is the 'live' one)
    
    # Structural Expectations
    suite.add_expectation(gxe.ExpectColumnToExist(column="transaction_id"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="amount"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="event_timestamp"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="payer_id"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="currency"))
    
    # Type & Uniqueness
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="amount", min_value=0.01))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="transaction_id", type_="str"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="transaction_id"))
    suite.add_expectation(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="label_available_timestamp",
        column_B="event_timestamp")
    )
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(
        column="label_available_timestamp"
        )
    )

    # suite.add_expectation(gxe.ExpectColumnValuesToBeIncreasing(
    # column="event_timestamp", 
    # mostly=0.99  # Allows 1% noise for duplicate milliseconds 
    # ))

    # No need to "save" manually - modifying the context-owned suite persists it 
    # (depending on the context type, e.g., FileDataContext auto-persists changes to JSON)
    print(f"✅ Suite '{suite_name}' registered and built.")
    return suite

def build_business_logic_suite(context):
    suite_name = "business_logic"

    try:
        context.suites.delete(suite_name)
    except Exception:
        pass

    # 1. Register new suite with Context
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

    # 2. Add Expectations
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="amount", min_value=0, max_value=1000000
    ))
    
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="currency", value_set=['INR']
    ))
    
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="payer_id"))

    print(f"✅ Suite '{suite_name}' registered and built.")
    return suite

def main():
    # 1. Initialize Context (File-based is best for real projects to see the JSONs)
    # If you use get_context() without args, it might be Ephemeral. 
    # Ideally, point to a local folder to ensure persistence works.
    context = gx.get_context(context_root_dir="great_expectations")

    
    print("--- Building Expectation Suites (Context-Managed) ---")
    build_schema_suite(context)
    build_business_logic_suite(context)
    print("\n🎉 Suites successfully registered in Data Context.")

if __name__ == "__main__":
    main()