#!/bin/bash

# Demo script showing how to run the monitor and benchmark

echo "🚀 GPU Agents Benchmark Suite - Monitor Demo"
echo "==========================================="
echo ""
echo "The TUI monitor provides real-time visualization of benchmark progress."
echo ""
echo "To run the monitor and benchmark together:"
echo ""
echo "1. In Terminal 1 (Monitor):"
echo "   $ cd crates/gpu-agents"
echo "   $ cargo run --bin monitor"
echo ""
echo "2. In Terminal 2 (Benchmark):"
echo "   $ cd crates/gpu-agents"
echo "   $ cargo run --bin benchmark"
echo ""
echo "Monitor Features:"
echo "  • Tab/Shift+Tab - Switch between views"
echo "  • q/Esc - Quit"
echo "  • Real-time progress tracking"
echo "  • GPU/CPU/Memory monitoring"
echo "  • Log file viewing"
echo ""
echo "Creating sample progress log for demo..."

# Create a sample progress log to demonstrate
cat > benchmark_progress.log << 'EOF'
[2024-01-20 10:00:00] 🚀 Starting GPU Agents Benchmark Suite
[2024-01-20 10:00:01] 🔍 System Check - Validating GPU availability
[2024-01-20 10:00:02] ✓ Found NVIDIA GPU: RTX 4090
[2024-01-20 10:00:03] ✓ CUDA Version: 12.2
[2024-01-20 10:00:04] 📈 Phase 1/4 - Agent Scalability Tests
[2024-01-20 10:00:05] Testing 1K agents spawn rate...
[2024-01-20 10:00:06] Progress: 10%
[2024-01-20 10:00:07] 💻 GPU Usage: 45%
[2024-01-20 10:00:08] Testing 10K agents spawn rate...
[2024-01-20 10:00:09] Progress: 25%
[2024-01-20 10:00:10] 💻 GPU Usage: 78%
[2024-01-20 10:00:11] Testing 100K agents spawn rate...
[2024-01-20 10:00:12] Progress: 40%
[2024-01-20 10:00:13] 💻 GPU Usage: 92%
[2024-01-20 10:00:14] Testing 1M agents spawn rate...
[2024-01-20 10:00:15] Progress: 50%
[2024-01-20 10:00:16] 🧠 Phase 2/4 - LLM Integration Tests
[2024-01-20 10:00:17] Initializing LLM models...
[2024-01-20 10:00:18] Progress: 60%
[2024-01-20 10:00:19] Testing collective reasoning...
[2024-01-20 10:00:20] Progress: 70%
[2024-01-20 10:00:21] 🕸️ Phase 3/4 - Knowledge Graph Tests
[2024-01-20 10:00:22] Building knowledge graph...
[2024-01-20 10:00:23] Progress: 80%
[2024-01-20 10:00:24] 🧬 Phase 4/4 - Evolution Strategy Tests
[2024-01-20 10:00:25] Running evolution algorithms...
[2024-01-20 10:00:26] Progress: 90%
[2024-01-20 10:00:27] 📊 Report Generation
[2024-01-20 10:00:28] Generating benchmark report...
[2024-01-20 10:00:29] Progress: 95%
[2024-01-20 10:00:30] ✅ Complete - All benchmarks finished successfully!
[2024-01-20 10:00:31] Progress: 100%
EOF

echo "✓ Sample log created: benchmark_progress.log"
echo ""
echo "Monitor would display:"
echo "┌─────────────────────────────────────────────────┐"
echo "│ GPU Agents Monitor                              │"
echo "├─────────────────────────────────────────────────┤"
echo "│ [Overview] Progress Resources Logs              │"
echo "├─────────────────────────────────────────────────┤"
echo "│ Phase                                           │"
echo "│ Current Phase: Evolution Strategy Tests         │"
echo "├─────────────────────────────────────────────────┤"
echo "│ Overall Progress                                │"
echo "│ ████████████████████████████████████░░░░ 90%   │"
echo "├─────────────────────────────────────────────────┤"
echo "│ Phase Progress                                  │"
echo "│ ██████████████████░░░░░░░░░░░░░░░░░░░░░ 45%   │"
echo "├─────────────────────────────────────────────────┤"
echo "│ Status                                          │"
echo "│ Current Operation: Running evolution algorithms │"
echo "└─────────────────────────────────────────────────┘"
echo ""

# Show what the benchmark command would look like
echo "Benchmark command examples:"
echo ""
echo "  # Quick benchmark (reduced scope)"
echo "  $ cargo run --bin benchmark -- --quick"
echo ""
echo "  # Full benchmark with custom output"
echo "  $ cargo run --bin benchmark -- --output reports/full_test"
echo ""
echo "  # Specific test only"
echo "  $ cargo run --bin benchmark -- --scalability-only"
echo ""
echo "  # Stress test (maximum scope)"
echo "  $ cargo run --bin benchmark -- --stress"
echo ""

# Check if we're in a terminal that supports TUI
if [ -t 0 ] && [ -t 1 ]; then
    echo "Press Enter to try running the monitor (requires terminal)..."
    read -r
    echo "Starting monitor with demo log file..."
    cargo run --bin monitor -- --log-file benchmark_progress.log
else
    echo "Note: TUI monitor requires an interactive terminal to run."
    echo "Please run this script in a proper terminal environment."
fi