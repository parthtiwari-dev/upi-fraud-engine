"""
Phase 4 Complete Validation Script
Location: src/features/tests/validate_phase4.py

Run from project root:
    python -m src.features.tests.validate_phase4

Or run directly:
    cd src/features/tests
    python validate_phase4.py
"""

import duckdb
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.schema import FeatureVector

# =============================================================================
# CONFIGURATION - Auto-detect paths relative to project root
# =============================================================================

# Auto-detect project root and construct paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Your database files
FEATURES_DB = str(PROCESSED_DIR / "features.duckdb")
FULL_FEATURES_DB = str(PROCESSED_DIR / "full_features.duckdb")
SOURCE_DB = str(RAW_DIR / "transactions_labeled.duckdb")

# Also check common alternate locations
ALTERNATE_PATHS = {
    'features': [
        str(PROCESSED_DIR / "step4_risk.duckdb"),
        str(PROCESSED_DIR / "step4_features.duckdb"),
    ],
    'full_features': [
        str(PROCESSED_DIR / "training_data.duckdb"),
        str(PROCESSED_DIR / "final_features.duckdb"),
    ],
    'source': [
        str(DATA_DIR / "transactions_labeled.duckdb"),
        str(RAW_DIR / "labeled_transactions.duckdb"),
    ]
}

# Expected schema from schema.py
EXPECTED_ENGINEERED_FEATURES = list(FeatureVector.model_fields.keys())

# =============================================================================
# PATH DETECTION
# =============================================================================

def find_database(primary_path: str, alternates: List[str], db_type: str) -> str:
    """Try to find the database file from primary or alternate paths"""
    if os.path.exists(primary_path):
        return primary_path
    
    for alt_path in alternates:
        if os.path.exists(alt_path):
            print(f"   Found {db_type} at alternate location: {alt_path}")
            return alt_path
    
    return primary_path  # Return primary even if not found (will fail later with clear error)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

class Colors:
    """Terminal colors for better readability"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """Print info message"""
    print(f"   {text}")

# =============================================================================
# TEST 1: File Existence & Path Detection
# =============================================================================

def test_files_exist() -> Tuple[bool, Dict[str, str]]:
    """Verify all required database files exist"""
    print_header("TEST 1: Checking File Existence & Path Detection")
    
    print_info(f"Project root: {PROJECT_ROOT}")
    print_info(f"Data directory: {DATA_DIR}\n")
    
    # Try to find databases
    features_db = find_database(FEATURES_DB, ALTERNATE_PATHS['features'], "features DB")
    full_db = find_database(FULL_FEATURES_DB, ALTERNATE_PATHS['full_features'], "full features DB")
    source_db = find_database(SOURCE_DB, ALTERNATE_PATHS['source'], "source DB")
    
    files_to_check = {
        "Engineered Features DB": features_db,
        "Full Features DB": full_db,
        "Source Labeled Data": source_db
    }
    
    all_exist = True
    found_paths = {}
    
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            rel_path = Path(path).relative_to(PROJECT_ROOT)
            print_success(f"{name}: {rel_path} ({size_mb:.1f} MB)")
            found_paths[name] = path
        else:
            rel_path = Path(path).relative_to(PROJECT_ROOT) if Path(path).is_relative_to(PROJECT_ROOT) else path
            print_error(f"{name} NOT FOUND: {rel_path}")
            all_exist = False
    
    if not all_exist:
        print_warning("\nTip: Place your database files in one of these locations:")
        print_info("  - data/processed/features.duckdb")
        print_info("  - data/processed/full_features.duckdb")
        print_info("  - data/raw/transactions_labeled.duckdb")
    
    return all_exist, found_paths

# =============================================================================
# TEST 2: Engineered Features Schema
# =============================================================================

def test_engineered_features_schema(db_path: str) -> Tuple[bool, Dict]:
    """Verify the engineered features database has correct schema"""
    print_header("TEST 2: Engineered Features Schema")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Get all tables
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            print_error("No tables found in features database!")
            con.close()
            return False, {}
        
        table_name = tables[0][0]
        print_info(f"Table name: {table_name}")
        
        # Get columns
        cols = con.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """).fetchall()
        
        actual_cols = [col[0] for col in cols]
        
        print_info(f"Expected columns: {len(EXPECTED_ENGINEERED_FEATURES)}")
        print_info(f"Actual columns: {len(actual_cols)}\n")
        
        # Check each expected column
        missing = []
        extra = []
        
        for col in EXPECTED_ENGINEERED_FEATURES:
            if col in actual_cols:
                print_success(f"Column '{col}' ✓")
            else:
                print_error(f"Column '{col}' MISSING")
                missing.append(col)
        
        for col in actual_cols:
            if col not in EXPECTED_ENGINEERED_FEATURES:
                print_warning(f"Extra column '{col}' (not in schema.py)")
                extra.append(col)
        
        # Get row count
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print_info(f"\nTotal rows: {row_count:,}")
        
        # Check for nulls in key columns
        null_check = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE transaction_id IS NULL) as null_txn_id,
                COUNT(*) FILTER (WHERE event_timestamp IS NULL) as null_timestamp
            FROM {table_name}
        """).fetchone()
        
        if null_check[0] > 0 or null_check[1] > 0:
            print_error(f"Found nulls: {null_check[0]} in transaction_id, {null_check[1]} in event_timestamp")
        else:
            print_success("No nulls in key columns")
        
        con.close()
        
        stats = {
            'table_name': table_name,
            'row_count': row_count,
            'column_count': len(actual_cols),
            'missing_columns': missing,
            'extra_columns': extra
        }
        
        success = len(missing) == 0 and len(actual_cols) == len(EXPECTED_ENGINEERED_FEATURES)
        return success, stats
        
    except Exception as e:
        print_error(f"Error checking schema: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

# =============================================================================
# TEST 3: Feature Value Ranges
# =============================================================================

def test_feature_ranges(db_path: str, table_name: str) -> bool:
    """Verify feature values are in valid ranges"""
    print_header("TEST 3: Feature Value Ranges & Statistics")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Check velocity features (should be >= 0)
        velocity_features = [
            'payer_txn_count_5min',
            'payer_txn_count_1h', 
            'payer_txn_count_24h',
            'device_txn_count_1h',
            'device_txn_count_24h'
        ]
        
        all_valid = True
        
        print_info("Velocity Features:")
        for feature in velocity_features:
            stats = con.execute(f"""
                SELECT 
                    MIN({feature}) as min_val,
                    MAX({feature}) as max_val,
                    AVG({feature}) as avg_val,
                    COUNT(*) FILTER (WHERE {feature} < 0) as negative_count
                FROM {table_name}
            """).fetchone()
            
            min_val, max_val, avg_val, neg_count = stats
            
            if neg_count > 0:
                print_error(f"  {feature}: Found {neg_count} negative values!")
                all_valid = False
            elif min_val >= 0:
                print_success(f"  {feature}: min={min_val}, max={max_val:.0f}, avg={avg_val:.2f}")
            else:
                print_error(f"  {feature}: Invalid range")
                all_valid = False
        
        # Check sum features (should be >= 0)
        print_info("\nAmount Sum Features:")
        sum_features = ['payer_txn_sum_5min', 'payer_txn_sum_1h', 'payer_txn_sum_24h']
        for feature in sum_features:
            stats = con.execute(f"""
                SELECT MIN({feature}), MAX({feature}), AVG({feature})
                FROM {table_name}
            """).fetchone()
            print_info(f"  {feature}: min=${stats[0]:.2f}, max=${stats[1]:.2f}, avg=${stats[2]:.2f}")
        
        # Check graph features
        print_info("\nGraph Features:")
        graph_stats = con.execute(f"""
            SELECT 
                AVG(device_distinct_payers_7d) as avg_device_users,
                MAX(device_distinct_payers_7d) as max_device_users,
                AVG(payer_distinct_payees_7d) as avg_payees,
                MAX(payer_distinct_payees_7d) as max_payees
            FROM {table_name}
        """).fetchone()
        
        print_info(f"  device_distinct_payers_7d: avg={graph_stats[0]:.2f}, max={graph_stats[1]}")
        print_info(f"  payer_distinct_payees_7d: avg={graph_stats[2]:.2f}, max={graph_stats[3]}")
        
        # Check risk history
        print_info("\nRisk History:")
        risk_stats = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE payer_past_fraud_count_30d > 0) as users_with_history,
                MAX(payer_past_fraud_count_30d) as max_fraud_count,
                SUM(payer_past_fraud_count_30d) as total_fraud_count
            FROM {table_name}
        """).fetchone()
        
        print_info(f"  Transactions with fraud history: {risk_stats[0]:,}")
        print_info(f"  Max past frauds for one user: {risk_stats[1]}")
        print_info(f"  Total past fraud references: {risk_stats[2]:,}")
        
        con.close()
        return all_valid
        
    except Exception as e:
        print_error(f"Error checking ranges: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# TEST 4: Time Correctness Sample Check
# =============================================================================

def test_time_correctness(db_path: str, table_name: str) -> bool:
    """Verify no obvious future leakage in features"""
    print_header("TEST 4: Point-in-Time Correctness (Sample Check)")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Sample transactions with velocity > 0
        sample = con.execute(f"""
            SELECT 
                transaction_id,
                event_timestamp,
                payer_txn_count_5min,
                payer_txn_sum_5min,
                payer_txn_count_1h
            FROM {table_name}
            WHERE payer_txn_count_5min > 0
            ORDER BY RANDOM()
            LIMIT 3
        """).fetchall()
        
        con.close()
        
        print_info("Sample transactions (checking for reasonable values):\n")
        
        all_reasonable = True
        for txn in sample:
            txn_id, ts, count_5m, sum_5m, count_1h = txn
            print_info(f"  {txn_id}:")
            print_info(f"    Time: {ts}")
            print_info(f"    5min: count={count_5m}, sum=${sum_5m:.2f}")
            print_info(f"    1hr: count={count_1h}")
            
            # Sanity checks
            if count_1h < count_5m:
                print_error(f"    ⚠️ 1h count < 5min count (impossible!)")
                all_reasonable = False
            
            print_info("")
        
        if all_reasonable:
            print_success("Sample values look reasonable")
            print_info("\nFor comprehensive time correctness validation, run:")
            print_info("  pytest src/features/tests/test_time_correctness.py -v")
        
        return all_reasonable
        
    except Exception as e:
        print_error(f"Error in time correctness check: {e}")
        return False

# =============================================================================
# TEST 5: Full Features Database
# =============================================================================

def test_full_features(db_path: str) -> Tuple[bool, Dict]:
    """Verify the full features database with raw columns"""
    print_header("TEST 5: Full Features Database (Engineered + Raw)")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        # Get table name
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            print_error("No tables found in full features database!")
            con.close()
            return False, {}
        
        table_name = tables[0][0]
        print_info(f"Table name: {table_name}")
        
        # Get row count
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Get column count
        col_count = con.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """).fetchone()[0]
        
        # Check fraud rate
        fraud_stats = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE is_fraud = 1) as fraud_count,
                COUNT(*) as total_count
            FROM {table_name}
        """).fetchone()
        
        fraud_count, total = fraud_stats
        fraud_rate = (fraud_count / total) * 100 if total > 0 else 0
        
        print_success(f"Rows: {row_count:,}")
        print_success(f"Columns: {col_count}")
        print_success(f"Fraud count: {fraud_count:,} ({fraud_rate:.2f}%)")
        
        # Verify engineered features are present
        print_info("\nChecking engineered features are present:")
        
        cols = con.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """).fetchall()
        
        actual_cols = [col[0] for col in cols]
        
        engineered_present = []
        engineered_missing = []
        
        for feature in EXPECTED_ENGINEERED_FEATURES:
            if feature in actual_cols:
                engineered_present.append(feature)
            else:
                engineered_missing.append(feature)
        
        if engineered_missing:
            print_error(f"  Missing {len(engineered_missing)} engineered features:")
            for feat in engineered_missing:
                print_error(f"    - {feat}")
        else:
            print_success(f"  All {len(EXPECTED_ENGINEERED_FEATURES)} engineered features present")
        
        # Check for raw Vesta columns
        vesta_prefixes = ['V', 'C', 'D', 'M', 'card', 'addr', 'dist', 'P_emaildomain', 'R_emaildomain']
        raw_vesta_cols = [c for c in actual_cols if any(c.startswith(prefix) for prefix in vesta_prefixes)]
        print_info(f"\nRaw Vesta columns: {len(raw_vesta_cols)}")
        print_info(f"Total engineered + raw: {len(engineered_present)} + {len(raw_vesta_cols)} = {len(engineered_present) + len(raw_vesta_cols)}")
        
        con.close()
        
        stats = {
            'row_count': row_count,
            'column_count': col_count,
            'fraud_count': fraud_count,
            'fraud_rate': fraud_rate,
            'engineered_features_present': len(engineered_present),
            'raw_columns': len(raw_vesta_cols)
        }
        
        # Success criteria
        success = (
            row_count > 0 and 
            len(engineered_missing) == 0 and
            2.0 < fraud_rate < 5.0 and
            col_count > 400  # Should have ~490 columns total
        )
        
        if not success:
            if fraud_rate < 2.0 or fraud_rate > 5.0:
                print_warning(f"Fraud rate {fraud_rate:.2f}% is outside expected range (3-4%)")
            if col_count <= 400:
                print_warning(f"Column count {col_count} seems low (expected ~490)")
        
        return success, stats
        
    except Exception as e:
        print_error(f"Error checking full features: {e}")
        import traceback
        traceback.print_exc()
        return False, {}

# =============================================================================
# TEST 6: Row Count Consistency
# =============================================================================

def test_row_consistency(paths: Dict[str, str], features_stats: Dict, full_stats: Dict) -> bool:
    """Verify row counts match across databases"""
    print_header("TEST 6: Row Count Consistency")
    
    try:
        # Source
        con_src = duckdb.connect(paths["Source Labeled Data"], read_only=True)
        src_rows = con_src.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        con_src.close()
        
        feat_rows = features_stats.get('row_count', 0)
        full_rows = full_stats.get('row_count', 0)
        
        print_info(f"Source DB:          {src_rows:,} rows")
        print_info(f"Features DB:        {feat_rows:,} rows")
        print_info(f"Full Features DB:   {full_rows:,} rows")
        
        if src_rows == feat_rows == full_rows:
            print_success("✓ All databases have matching row counts")
            return True
        else:
            print_error("Row count mismatch detected!")
            if src_rows != feat_rows:
                print_warning(f"  Source ({src_rows:,}) != Features ({feat_rows:,})")
            if src_rows != full_rows:
                print_warning(f"  Source ({src_rows:,}) != Full Features ({full_rows:,})")
            return False
        
    except Exception as e:
        print_error(f"Error checking row consistency: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all validation tests"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         PHASE 4 VALIDATION - Feature Engineering Pipeline         ║")
    print("║                 Point-in-Time Safe Feature Store                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    results = {}
    
    # Test 1: File existence
    files_ok, found_paths = test_files_exist()
    results['files_exist'] = files_ok
    
    if not files_ok:
        print_error("\n⚠️  Required files are missing.")
        print_info("\nExpected locations:")
        print_info(f"  {Path(FEATURES_DB).relative_to(PROJECT_ROOT)}")
        print_info(f"  {Path(FULL_FEATURES_DB).relative_to(PROJECT_ROOT)}")
        print_info(f"  {Path(SOURCE_DB).relative_to(PROJECT_ROOT)}")
        return
    
    # Test 2: Schema
    features_path = found_paths["Engineered Features DB"]
    results['schema'], schema_stats = test_engineered_features_schema(features_path)
    
    if not results['schema']:
        print_warning("\n⚠️  Schema validation failed. Subsequent tests may fail.")
    
    # Test 3: Ranges
    table_name = schema_stats.get('table_name', 'features')
    results['ranges'] = test_feature_ranges(features_path, table_name)
    
    # Test 4: Time correctness
    results['time_correctness'] = test_time_correctness(features_path, table_name)
    
    # Test 5: Full features
    full_path = found_paths["Full Features DB"]
    results['full_features'], full_stats = test_full_features(full_path)
    
    # Test 6: Row consistency
    results['row_consistency'] = test_row_consistency(found_paths, schema_stats, full_stats)
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"{status:20} {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print(f"{Colors.BOLD}{Colors.GREEN}")
        print("🎉 ALL TESTS PASSED - Phase 4 is ready for Phase 5!")
        print(f"{Colors.END}")
        print("\nNext steps:")
        print("  1. Run unit tests:")
        print("     pytest src/features/tests/test_time_correctness.py -v")
        print("  2. Proceed to Phase 5: Model Training")
        print("\nYour datasets:")
        print(f"  - Features: {schema_stats.get('row_count', 0):,} rows, {schema_stats.get('column_count', 0)} columns")
        print(f"  - Training: {full_stats.get('row_count', 0):,} rows, {full_stats.get('column_count', 0)} columns")
        print(f"  - Fraud rate: {full_stats.get('fraud_rate', 0):.2f}%")
    else:
        print(f"{Colors.BOLD}{Colors.RED}")
        print("❌ SOME TESTS FAILED - Please review errors above")
        print(f"{Colors.END}")
        print("\nRecommended actions:")
        print("  1. Check that database files are in correct locations")
        print("  2. Re-run feature engineering pipeline if schema is incorrect")
        print("  3. Review error messages for specific issues")
        print("  4. Check the README for expected outputs")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()