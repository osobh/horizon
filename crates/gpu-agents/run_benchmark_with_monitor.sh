#!/bin/bash

# GPU Agents Benchmark with TUI Monitor
# This script launches both the benchmark and monitor in a tmux session

SESSION_NAME="gpu-benchmark"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Split window horizontally (monitor on top, benchmark on bottom)
tmux split-window -v -t $SESSION_NAME

# Run monitor in top pane
tmux send-keys -t $SESSION_NAME:0.0 'cargo run --bin monitor' Enter

# Give monitor time to start
sleep 2

# Run benchmark in bottom pane
tmux send-keys -t $SESSION_NAME:0.1 'cargo run --bin benchmark' Enter

# Attach to session
tmux attach-session -t $SESSION_NAME