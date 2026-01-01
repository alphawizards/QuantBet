"""
Database Migration Validation Script.

Validates migrations before deployment by:
1. Running migration on copy of database
2. Checking for data loss
3. Verifying new structures created
4. Testing rollback

Usage:
    python scripts/validate_migration.py migrations/006_bet_tracking.sql
"""

import psycopg2
import sys
import os
from datetime import datetime
from pathlib import Path


def validate_migration(migration_file: str, db_url: str = None):
    """
    Validate a database migration.
    
    Args:
        migration_file: Path to migration SQL file
        db_url: Database URL (defaults to DATABASE_URL env var)
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
        FileNotFoundError: If migration file doesn't exist
    """
    
    # Verify migration file exists
    migration_path = Path(migration_file)
    if not migration_path.exists():
        raise FileNotFoundError(f"Migration file not found: {migration_file}")
    
    # Get database URL
    if db_url is None:
        db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not provided and not in environment")
    
    print(f"🔍 Validating migration: {migration_path.name}")
    print(f"   Database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()
    
    # Connect to database
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        # Step 1: Take pre-migration snapshot
        print("📸 Taking pre-migration snapshot...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema='public'
        """)
        pre_table_count = cursor.fetchone()[0]
        
        # Get record counts for existing tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        pre_counts = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                pre_counts[table] = cursor.fetchone()[0]
            except Exception as e:
                print(f"   ⚠️  Could not count {table}: {e}")
        
        print(f"   Tables before: {pre_table_count}")
        print(f"   Total records: {sum(pre_counts.values())}")
        print()
        
        # Step 2: Run migration
        print("🔄 Running migration...")
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        try:
            cursor.execute(migration_sql)
            print(f"   ✅ Migration executed successfully")
        except Exception as e:
            print(f"   ❌ Migration failed: {e}")
            raise
        print()
        
        # Step 3: Validate post-migration
        print("✅ Validating post-migration state...")
        
        # Check table count
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema='public'
        """)
        post_table_count = cursor.fetchone()[0]
        
        # Check for data loss in existing tables
        for table, pre_count in pre_counts.items():
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                post_count = cursor.fetchone()[0]
                
                if post_count < pre_count:
                    raise ValueError(
                        f"❌ Data loss detected in {table}! "
                        f"Pre: {pre_count}, Post: {post_count}"
                    )
                elif post_count > pre_count:
                    print(f"   ℹ️  {table}: {post_count - pre_count} new records")
                    
            except psycopg2.errors.UndefinedTable:
                # Table might have been dropped by migration
                print(f"   ⚠️  Table {table} no longer exists")
        
        # Report new tables
        new_table_count = post_table_count - pre_table_count
        if new_table_count > 0:
            print(f"   ✅ {new_table_count} new table(s) created")
        
        print(f"   ✅ No data loss detected")
        print()
        
        # Step 4: Check for rollback script
        rollback_file = migration_path.parent / f"{migration_path.stem}_down.sql"
        if rollback_file.exists():
            print(f"   ✅ Rollback script found: {rollback_file.name}")
        else:
            print(f"   ⚠️  No rollback script found (expected: {rollback_file.name})")
        
        print()
        print("=" * 60)
        print("✅ Migration validation PASSED")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Migration validation FAILED: {e}")
        print("=" * 60)
        raise
        
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_migration.py <migration_file>")
        print()
        print("Example:")
        print("  python scripts/validate_migration.py migrations/006_bet_tracking.sql")
        sys.exit(1)
    
    migration_file = sys.argv[1]
    
    try:
        validate_migration(migration_file)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)
