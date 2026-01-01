#!/bin/bash
# Teardown Test Database for QuantBet
# Usage: ./scripts/teardown_test_db.sh

set -e

echo "🧹 Tearing down QuantBet test database..."

# Configuration
TEST_DB_NAME="quantbet_test"
TEST_DB_USER="${TEST_DB_USER:-postgres}"

# Drop test database
echo "📦 Dropping test database: $TEST_DB_NAME..."
dropdb --if-exists $TEST_DB_NAME -U $TEST_DB_USER || {
    echo "⚠️  Database $TEST_DB_NAME does not exist or already dropped"
}

echo "✅ Test database teardown complete!"
