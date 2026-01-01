"""
Test database connection and check schema status.
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

print(f"Testing connection to: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Test connection
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print(f"✓ Connected to PostgreSQL")
        print(f"  Version: {version[:50]}...")
        
        # Check if tables exist
        result = conn.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename;
        """))
        
        tables = [row[0] for row in result.fetchall()]
        
        print(f"\n✓ Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        # Check for migration tables
        migration_tables = [
            'backtest_validations',
            'backtest_window_results',
            'parameter_sensitivity_results',
            'calibration_history',
            'drift_baselines',
            'drift_alerts',
            'performance_alerts',
            'retraining_jobs'
        ]
        
        missing_tables = [t for t in migration_tables if t not in tables]
        
        if missing_tables:
            print(f"\n⚠ Missing migration tables ({len(missing_tables)}):")
            for table in missing_tables:
                print(f"  - {table}")
            print("\n→ Need to apply migrations")
        else:
            print(f"\n✓ All migration tables present!")
        
        # Check data counts
        print(f"\n📊 Data counts:")
        for table in tables[:5]:  # Check first 5 tables
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                count = result.fetchone()[0]
                print(f"  {table}: {count} rows")
            except Exception as e:
                print(f"  {table}: Error - {e}")
        
except Exception as e:
    print(f"✗ Connection failed:")
    print(f"  {type(e).__name__}: {e}")
    print(f"\n→ Check if PostgreSQL is running")
    print(f"→ Check credentials in .env")
