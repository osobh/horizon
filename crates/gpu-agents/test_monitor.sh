#!/bin/bash

# Test the monitor with a long-running benchmark

echo "Starting monitor in the background..."
cargo run --bin monitor &
MONITOR_PID=$!

sleep 2

echo "Starting benchmark (stress test)..."
cargo run --bin benchmark -- --stress

echo "Benchmark complete. Press Ctrl+C to stop the monitor."
wait $MONITOR_PID