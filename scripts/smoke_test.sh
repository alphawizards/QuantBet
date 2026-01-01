#!/bin/bash
# Smoke Test Suite - Run after every deployment
# Must complete in < 2 minutes

set -e

echo "🧪 Starting QuantBet Smoke Tests..."
echo "====================================="

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:3000}"
TIMEOUT=10

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

failed_tests=0
total_tests=0

# Test function
run_test() {
    local test_name=$1
    local command=$2
    
    total_tests=$((total_tests + 1))
    echo -n "Testing: $test_name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL${NC}"
        failed_tests=$((failed_tests + 1))
    fi
}

# API Health Check
run_test "API Health Endpoint" \
    "curl -f -s --max-time $TIMEOUT $API_URL/health | grep -q '\"status\":\"healthy\"'"

# Predictions Endpoint
run_test "Predictions Endpoint" \
    "curl -f -s --max-time $TIMEOUT $API_URL/predictions/today | grep -q '\['"

# Database Connection (via health endpoint)
run_test "Database Connection" \
    "curl -f -s --max-time $TIMEOUT $API_URL/health | grep -q '\"database_connected\":true'"

# Frontend Loads
run_test "Frontend Homepage" \
    "curl -f -s --max-time $TIMEOUT $DASHBOARD_URL | grep -q 'QuantBet'"

# API Response Time
run_test "API Response Time < 2s" \
    "[[ \$(curl -w '%{time_total}' -s -o /dev/null $API_URL/health) < 2 ]]"

echo "====================================="
echo "Smoke Tests Complete: $((total_tests - failed_tests))/$total_tests passed"

if [ $failed_tests -gt 0 ]; then
    echo -e "${RED}❌ SMOKE TESTS FAILED${NC}"
    echo "Failed tests: $failed_tests"
    exit 1
else
    echo -e "${GREEN}✅ ALL SMOKE TESTS PASSED${NC}"
    exit 0
fi
