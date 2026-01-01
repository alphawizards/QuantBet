#!/bin/bash
# Setup Test Database for QuantBet
# Usage: ./scripts/setup_test_db.sh

set -e

echo "🔧 Setting up QuantBet test database..."

# Configuration
export TEST_DB_NAME="quantbet_test"
export TEST_DB_USER="${TEST_DB_USER:-postgres}"
export TEST_DB_PASSWORD="${TEST_DB_PASSWORD:-postgres}"
export TEST_DB_HOST="${TEST_DB_HOST:-localhost}"
export TEST_DB_PORT="${TEST_DB_PORT:-5432}"

export DATABASE_URL="postgresql://${TEST_DB_USER}:${TEST_DB_PASSWORD}@${TEST_DB_HOST}:${TEST_DB_PORT}/${TEST_DB_NAME}"

# Drop existing test database if exists
echo "📦 Dropping existing test database (if exists)..."
dropdb --if-exists $TEST_DB_NAME -U $TEST_DB_USER || true

# Create new test database
echo "📦 Creating test database: $TEST_DB_NAME..."
createdb $TEST_DB_NAME -U $TEST_DB_USER

# Run migrations
echo "🔄 Running database migrations..."
python scripts/apply_migrations.py || {
    echo "❌ Migrations failed!"
    exit 1
}

# Seed test data
echo "🌱 Seeding test data..."
python scripts/seed_test_data.py || {
    echo "❌ Seeding failed!"
    exit 1
}

# Verify database setup
echo "✅ Verifying database setup..."
psql $DATABASE_URL -c "SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema='public';" || {
    echo "❌ Database verification failed!"
    exit 1
}

echo "✅ Test database setup complete!"
echo "   Database: $TEST_DB_NAME"
echo "   URL: $DATABASE_URL"
