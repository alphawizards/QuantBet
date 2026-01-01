"""
Apply remaining migrations with fresh connections.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "database" / "schema.sql" / "migrations"

print("=" * 60)
print("Applying Remaining Migrations (003-005)")
print("=" * 60)

MIGRATIONS = [
    "003_parameter_sensitivity.sql",
    "004_transaction_costs.sql",
    "005_monitoring_tables.sql"
]

for migration_file in MIGRATIONS:
    migration_path = MIGRATIONS_DIR / migration_file
    
    print(f"\n📝 {migration_file}")
    
    if not migration_path.exists():
        print(f"  ⚠ File not found")
        continue
    
    sql_content = migration_path.read_text()
    
    # Create fresh engine and connection for each migration
    try:
        engine = create_engine(DATABASE_URL)
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(sql_content))
        conn.close()
        engine.dispose()
        print(f"  ✓ Applied successfully!")
        
    except Exception as e:
        error_msg = str(e)
        if "already exists" in error_msg.lower():
            print(f"  ℹ Already applied")
        else:
            print(f"  ✗ Error: {error_msg[:200]}")
            # Continue anyway
            try:
                conn.close()
                engine.dispose()
            except:
                pass

# Final verification
print(f"\n" + "=" * 60)
print("Final Verification")
print("=" * 60)

engine = create_engine(DATABASE_URL)
conn = engine.connect()

result = conn.execute(text("""
    SELECT table_name
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
    )
    ORDER BY table_name;
"""))

tables = [row[0] for row in result.fetchall()]
conn.close()

print(f"\n✓ Migration tables: {len(tables)}/8")
for table in tables:
    print(f"  ✓ {table}")

if len(tables) < 8:
    missing = set(['backtest_validations', 'backtest_window_results', 'parameter_sensitivity_results',
                   'calibration_history', 'drift_baselines', 'drift_alerts', 
                   'performance_alerts', 'retraining_jobs']) - set(tables)
    print(f"\n⚠ Missing:")
    for table in missing:
        print(f"  ✗ {table}")

print("\n" + "=" * 60)
