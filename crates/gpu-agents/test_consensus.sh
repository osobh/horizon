#!/bin/bash
# Quick test of consensus benchmark

echo "🧪 Testing GPU Consensus Performance"
echo "===================================="

# Build in release mode
echo "Building consensus benchmark..."
cargo build --release --bin consensus-benchmark

# Test with different agent counts
for agents in 100 1000 10000; do
    echo -e "\n📊 Testing with $agents agents..."
    cargo run --release --bin consensus-benchmark -- \
        --agents $agents \
        --iterations 100 \
        --byzantine-percent 0 \
        --output "consensus_${agents}_agents.json" \
        --warmup 20
done

echo -e "\n✅ Test complete!"
echo "Check consensus_*_agents.json files for detailed results"