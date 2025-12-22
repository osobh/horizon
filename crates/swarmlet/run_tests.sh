#!/bin/bash
# Script to run swarmlet tests and show coverage

echo "Running Swarmlet Unit Tests..."
echo "=============================="

cd "$(dirname "$0")"

# Run tests with coverage if available
if command -v cargo-tarpaulin &> /dev/null; then
    echo "Running with coverage using cargo-tarpaulin..."
    cargo tarpaulin --lib --out Stdout --exclude-files "*/tests/*" --exclude-files "*/examples/*"
else
    echo "Running tests without coverage (install cargo-tarpaulin for coverage)..."
    cargo test --lib -- --nocapture
fi

# Count tests
echo ""
echo "Test Summary:"
echo "============="
TOTAL_TESTS=$(cargo test --lib -- --list 2>/dev/null | grep "test " | wc -l)
echo "Total unit tests: $TOTAL_TESTS"

# Show test modules
echo ""
echo "Test Modules:"
echo "============"
echo "- agent: $(cargo test agent:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- config: $(cargo test config:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- discovery: $(cargo test discovery:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- join: $(cargo test join:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- profile: $(cargo test profile:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- security: $(cargo test security:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"
echo "- workload: $(cargo test workload:: --lib -- --list 2>/dev/null | grep "test " | wc -l) tests"

echo ""
echo "All tests completed!"