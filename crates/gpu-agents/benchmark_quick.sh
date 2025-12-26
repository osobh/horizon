#!/bin/bash

# Quick benchmark runner with monitor

echo "🚀 Starting GPU Agents Benchmark with Monitor"
echo "============================================"

# Start monitor in background
echo "Starting monitor dashboard..."
cargo run --bin monitor &
MONITOR_PID=$!

# Give monitor time to initialize
sleep 2

# Run benchmark
echo "Starting benchmark suite..."
cargo run --bin benchmark -- --quick

# When benchmark completes, kill monitor
echo "Benchmark complete. Press any key to stop monitor..."
read -n 1
kill $MONITOR_PID 2>/dev/null

echo "Done!"