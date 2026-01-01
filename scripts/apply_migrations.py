"""
Apply database migrations with correct connection handling.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "database" / "schema.sql" / "migrations"

print("=" * 60)
print("Database Migration Application")
print("=" * 60)

MIGRATIONS = [
    "002_walk_forward_validation.sql",
    "003_parameter_sensitivity.sql",
    "004_transaction_costs.sql",
    "005_monitoring_tables.sql"
]

engine = create_engine(DATABASE_URL)

for migration_file in MIGRATIONS:
    migration_path = MIGRATIONS_DIR / migration_file
    
    print(f"\n📝 {migration_file}")
    
    if not migration_path.exists():
        print(f"  ⚠ File not found: {migration_path}")
        continue
    
    sql_content = migration_path.read_text()
    
    try:
        # Use connection with autocommit for DDL
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(sql_content))
        conn.close()
        print(f"  ✓ Applied successfully!")
        
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            print(f"  ℹ Already applied")
        else:
            print(f"  ✗ Error: {error_msg[:150]}")

# Verify
print(f"\n" + "=" * 60)
print("Verification")
print("=" * 60)

conn = engine.connect()
result = conn.execute(text("""
    SELECT COUNT(*) as cnt
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN (
        'backtest_validations',
        'backtest_window_results',
        'parameter_sensitivity_results',
        'calibration_history',
        'drift_baselines',
        'drift_alerts',
        'performance_alerts',
        'retraining_jobs'
    );
"""))
count = result.fetchone()[0]
conn.close()

print(f"\n✓ Migration tables created: {count}/8")

if count == 8:
    print(f"🎉 All migrations applied successfully!")
else:
    print(f"⚠ Only {count}/8 tables created. Check errors above.")

print("=" * 60)
