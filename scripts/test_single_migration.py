"""
Apply ONE migration with detailed error reporting.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import traceback

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "database" / "schema.sql" / "migrations"

migration_file = "002_walk_forward_validation.sql"
migration_path = MIGRATIONS_DIR / migration_file

print(f"Testing single migration: {migration_file}")
print(f"Path: {migration_path}")
print(f"File exists: {migration_path.exists()}")

if not migration_path.exists():
    print(f"ERROR: File not found!")
    exit(1)

sql_content = migration_path.read_text()
print(f"\nSQL content length: {len(sql_content)} bytes")
print(f"First 200 chars:\n{sql_content[:200]}")

engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")

try:
    with engine.raw_connection() as conn:
        cursor = conn.cursor()
        try:
            print(f"\nExecuting SQL...")
            cursor.execute(sql_content)
            print(f"✓ Success!")
        except Exception as e:
            print(f"\n✗ ERROR:")
            print(f"Type: {type(e).__name__}")
            print(f"Message: {str(e)}")
            print(f"\nFull traceback:")
            traceback.print_exc()
        finally:
            cursor.close()
except Exception as e:
    print(f"\n✗ Connection ERROR:")
    traceback.print_exc()
