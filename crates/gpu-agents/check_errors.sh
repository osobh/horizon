#!/bin/bash
cd /home/osobh/projects/exorust/crates/gpu-agents
cargo check 2>&1 | tee build_output.txt
echo "Build complete, check build_output.txt"