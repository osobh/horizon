#!/bin/bash

# GPU Agents Benchmark Runner
# Comprehensive script to run GPU agent benchmarks with reproducible results

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEFAULT_OUTPUT_DIR="reports"
DEFAULT_CONFIG_FILE=""
BENCHMARK_TYPE="standard"
VERBOSE=false
CLEAN_BUILD=false
SKIP_DEPS=false
GENERATE_CHARTS=true
UPLOAD_RESULTS=false
PROGRESS_LOG="benchmark_progress.log"

# Function to print colored output
print_status() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to show usage
show_help() {
    cat << EOF
GPU Agents Benchmark Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -o, --output DIR        Output directory for results (default: $DEFAULT_OUTPUT_DIR)
    -c, --config FILE       Configuration file path
    -t, --type TYPE         Benchmark type: quick|standard|stress (default: standard)
    -v, --verbose           Enable verbose output
    --clean                 Clean build before running benchmarks
    --skip-deps             Skip dependency installation
    --no-charts            Skip chart generation
    --upload               Upload results to remote storage
    --scalability-only     Run only scalability benchmarks
    --llm-only             Run only LLM integration benchmarks
    --kg-only              Run only knowledge graph benchmarks
    --evolution-only       Run only evolution strategy benchmarks

EXAMPLES:
    # Run standard benchmarks
    $0

    # Run quick benchmarks with custom output directory
    $0 --type quick --output /tmp/quick_benchmark

    # Run stress test with verbose output
    $0 --type stress --verbose --clean

    # Run only scalability benchmarks
    $0 --scalability-only --output scalability_results

    # Run with custom configuration
    $0 --config benchmark_config.json --upload

BENCHMARK TYPES:
    quick     - Fast benchmark with reduced scope (5-10 minutes)
    standard  - Comprehensive benchmark suite (30-60 minutes)
    stress    - Maximum stress testing (1-3 hours)

ENVIRONMENT VARIABLES:
    CUDA_VISIBLE_DEVICES  - GPU devices to use (default: 0)
    BENCHMARK_TIMEOUT     - Maximum benchmark runtime in seconds
    BENCHMARK_LOG_LEVEL   - Log level: debug|info|warn|error
EOF
}

# Function to check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    # Check if CUDA is available
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. CUDA drivers may not be installed."
        exit 1
    fi
    
    # Check GPU availability
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpu_count" -eq 0 ]; then
        print_error "No CUDA-capable GPUs found."
        exit 1
    fi
    
    print_success "Found $gpu_count CUDA-capable GPU(s)"
    
    # Check if Rust is available
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust."
        exit 1
    fi
    
    # Check available memory
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 8 ]; then
        print_warning "System has less than 8GB RAM. Some benchmarks may fail."
    fi
    
    print_success "System requirements check passed"
}

# Function to install dependencies
install_dependencies() {
    if [ "$SKIP_DEPS" = true ]; then
        print_status "Skipping dependency installation"
        return
    fi
    
    print_step "Installing dependencies..."
    
    # Install Rust dependencies
    cargo fetch
    
    # Install additional tools if needed
    if ! command -v tokio-console &> /dev/null && [ "$VERBOSE" = true ]; then
        print_status "Installing tokio-console for async profiling..."
        cargo install tokio-console || print_warning "Failed to install tokio-console"
    fi
    
    print_success "Dependencies installed"
}

# Function to build the benchmark binary
build_benchmark() {
    print_step "Building benchmark binary..."
    
    if [ "$CLEAN_BUILD" = true ]; then
        print_status "Cleaning previous build..."
        cargo clean
    fi
    
    local build_flags=""
    if [ "$BENCHMARK_TYPE" = "stress" ]; then
        build_flags="--release"
        print_status "Building in release mode for stress testing"
    elif [ "$BENCHMARK_TYPE" = "quick" ]; then
        print_status "Building in debug mode for quick testing"
    else
        build_flags="--release"
        print_status "Building in release mode"
    fi
    
    if [ "$VERBOSE" = true ]; then
        cargo build $build_flags --bin benchmark
    else
        cargo build $build_flags --bin benchmark > /dev/null 2>&1
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Build failed"
        exit 1
    fi
    
    print_success "Benchmark binary built successfully"
}

# Function to setup output directory
setup_output_directory() {
    local base_output_dir="$1"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local full_output_dir="${base_output_dir}/benchmark_${timestamp}"
    
    print_step "Setting up output directory: $full_output_dir"
    
    # Create the base reports directory if it doesn't exist
    mkdir -p "$base_output_dir"
    
    # Create timestamped subdirectory
    mkdir -p "$full_output_dir"
    
    # Create subdirectories
    mkdir -p "$full_output_dir/logs"
    mkdir -p "$full_output_dir/charts"
    mkdir -p "$full_output_dir/raw_data"
    mkdir -p "$full_output_dir/configs"
    
    # Copy configuration files
    if [ -f "Cargo.toml" ]; then
        cp Cargo.toml "$full_output_dir/configs/"
    fi
    
    if [ -n "$DEFAULT_CONFIG_FILE" ] && [ -f "$DEFAULT_CONFIG_FILE" ]; then
        cp "$DEFAULT_CONFIG_FILE" "$full_output_dir/configs/"
    fi
    
    # Create system info file
    create_system_info "$full_output_dir"
    
    # Create/update 'latest' symlink to point to this run
    local latest_link="${base_output_dir}/latest"
    if [ -L "$latest_link" ]; then
        rm "$latest_link"
    fi
    ln -sf "benchmark_${timestamp}" "$latest_link"
    
    print_success "Created 'latest' symlink: $latest_link -> benchmark_${timestamp}"
    
    echo "$full_output_dir"
}

# Function to create system information file
create_system_info() {
    local output_dir="$1"
    local system_info_file="$output_dir/system_info.txt"
    
    print_status "Gathering system information..."
    
    {
        echo "GPU Agents Benchmark System Information"
        echo "======================================"
        echo "Timestamp: $(date)"
        echo "Hostname: $(hostname)"
        echo "User: $(whoami)"
        echo ""
        echo "SYSTEM INFORMATION:"
        echo "OS: $(uname -a)"
        echo "CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
        echo "Total RAM: $(free -h | awk '/^Mem:/ {print $2}')"
        echo "Available RAM: $(free -h | awk '/^Mem:/ {print $7}')"
        echo ""
        echo "GPU INFORMATION:"
        nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader,nounits
        echo ""
        echo "CUDA INFORMATION:"
        nvcc --version 2>/dev/null || echo "NVCC not available"
        echo ""
        echo "RUST INFORMATION:"
        rustc --version
        cargo --version
        echo ""
        echo "ENVIRONMENT VARIABLES:"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
        echo "BENCHMARK_TIMEOUT: ${BENCHMARK_TIMEOUT:-not set}"
        echo "BENCHMARK_LOG_LEVEL: ${BENCHMARK_LOG_LEVEL:-not set}"
        echo ""
        echo "BENCHMARK CONFIGURATION:"
        echo "Type: $BENCHMARK_TYPE"
        echo "Verbose: $VERBOSE"
        echo "Generate Charts: $GENERATE_CHARTS"
        echo "Config File: ${DEFAULT_CONFIG_FILE:-none}"
    } > "$system_info_file"
    
    print_success "System information saved to $system_info_file"
}

# Function to run benchmarks
run_benchmarks() {
    local output_dir="$1"
    local log_file="$output_dir/logs/benchmark.log"
    local progress_log="$PROGRESS_LOG"
    
    print_step "Running GPU Agents Benchmarks..."
    print_status "Benchmark type: $BENCHMARK_TYPE"
    print_status "Output directory: $output_dir"
    print_status "Log file: $log_file"
    print_status "Progress log: $progress_log"
    
    # Initialize progress log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting GPU Agents Benchmark Suite" > "$progress_log"
    
    # Prepare benchmark command
    local benchmark_cmd="./target/release/benchmark"
    if [ "$BENCHMARK_TYPE" = "quick" ] || [ "$BENCHMARK_TYPE" = "debug" ]; then
        benchmark_cmd="./target/debug/benchmark"
    fi
    
    local benchmark_args="--output $output_dir"
    
    # Add benchmark type flag
    case "$BENCHMARK_TYPE" in
        "quick")
            benchmark_args="$benchmark_args --quick"
            ;;
        "stress")
            benchmark_args="$benchmark_args --stress"
            ;;
    esac
    
    # Add configuration file if specified
    if [ -n "$DEFAULT_CONFIG_FILE" ]; then
        benchmark_args="$benchmark_args --config $DEFAULT_CONFIG_FILE"
    fi
    
    # Add verbose flag if requested
    if [ "$VERBOSE" = true ]; then
        benchmark_args="$benchmark_args --verbose"
    fi
    
    # Add specific benchmark flags if set
    if [ "$SCALABILITY_ONLY" = true ]; then
        benchmark_args="$benchmark_args --scalability-only"
    fi
    if [ "$LLM_ONLY" = true ]; then
        benchmark_args="$benchmark_args --llm-only"
    fi
    if [ "$KG_ONLY" = true ]; then
        benchmark_args="$benchmark_args --kg-only"
    fi
    if [ "$EVOLUTION_ONLY" = true ]; then
        benchmark_args="$benchmark_args --evolution-only"
    fi
    
    # Disable charts if requested
    if [ "$GENERATE_CHARTS" = false ]; then
        benchmark_args="$benchmark_args --no-reports"
    fi
    
    print_status "Running command: $benchmark_cmd $benchmark_args"
    
    # Set timeout if specified
    local timeout_cmd=""
    if [ -n "$BENCHMARK_TIMEOUT" ]; then
        timeout_cmd="timeout $BENCHMARK_TIMEOUT"
        print_status "Benchmark timeout: ${BENCHMARK_TIMEOUT}s"
    fi
    
    # Initialize progress monitoring
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting GPU Agents Benchmark Suite" > "$progress_log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔍 Checking system requirements and initializing" >> "$progress_log"
    
    # Log benchmark phases
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📈 Phase 1/4 - Agent Scalability Tests (Testing spawn rates and memory usage)" >> "$progress_log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧠 Phase 2/4 - LLM Integration Tests (Batch processing and throughput)" >> "$progress_log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🕸️  Phase 3/4 - Knowledge Graph Tests (Node scaling and query performance)" >> "$progress_log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧬 Phase 4/4 - Evolution Strategy Tests (Population dynamics and convergence)" >> "$progress_log"
    
    # Run the benchmark
    local start_time=$(date +%s)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚡ Execution - Running benchmark binary with progress monitoring" >> "$progress_log"
    
    if [ "$VERBOSE" = true ]; then
        # For verbose mode, capture and forward progress
        $timeout_cmd $benchmark_cmd $benchmark_args 2>&1 | tee "$log_file" | while IFS= read -r line; do
            # Forward specific progress lines to our progress log
            if echo "$line" | grep -E "(🚀|🔍|📈|🧠|🕸️|🧬|📊|✅|Phase|Testing|Benchmarking|Evaluating)" > /dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" >> "$progress_log"
            fi
            
            # Extract completion percentages if available
            if echo "$line" | grep -E "[0-9]+\.[0-9]+%|[0-9]+%" > /dev/null; then
                local percentage=$(echo "$line" | grep -oE "[0-9]+\.[0-9]+%|[0-9]+%" | head -1)
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 Progress: $percentage" >> "$progress_log"
            fi
        done
    else
        $timeout_cmd $benchmark_cmd $benchmark_args > "$log_file" 2>&1 &
        local benchmark_pid=$!
        
        # Monitor benchmark progress in background
        # Monitor benchmark progress with resource usage
        local update_count=0
        while kill -0 $benchmark_pid 2>/dev/null; do
            sleep 10
            update_count=$((update_count + 1))
            
            # Add periodic progress updates with resource info
            local current_phase="Unknown"
            if [ $update_count -le 3 ]; then
                current_phase="Scalability Tests"
            elif [ $update_count -le 6 ]; then
                current_phase="LLM Integration Tests"
            elif [ $update_count -le 9 ]; then
                current_phase="Knowledge Graph Tests"
            else
                current_phase="Evolution Strategy Tests"
            fi
            
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏳ $current_phase in progress... (Update #$update_count)" >> "$progress_log"
            
            # Add resource monitoring if nvidia-smi is available
            if command -v nvidia-smi &> /dev/null; then
                local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
                local gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
                if [ -n "$gpu_util" ] && [ -n "$gpu_mem" ]; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 💻 GPU Usage: ${gpu_util}% | Memory: ${gpu_mem}" >> "$progress_log"
                fi
            fi
        done
        
        wait $benchmark_pid
    fi
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Benchmark execution completed successfully in ${duration}s" >> "$progress_log"
        print_success "Benchmarks completed successfully in ${duration}s"
    elif [ $exit_code -eq 124 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏰ Benchmark execution timed out after ${BENCHMARK_TIMEOUT}s" >> "$progress_log"
        print_error "Benchmarks timed out after ${BENCHMARK_TIMEOUT}s"
        return 1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Benchmark execution failed with exit code $exit_code" >> "$progress_log"
        print_error "Benchmarks failed with exit code $exit_code"
        print_status "Check log file for details: $log_file"
        return 1
    fi
    
    return 0
}

# Function to generate additional charts
generate_additional_charts() {
    local output_dir="$1"
    
    if [ "$GENERATE_CHARTS" = false ]; then
        return
    fi
    
    print_step "Generating additional performance charts..."
    
    # Check if Python is available for additional chart generation
    if command -v python3 &> /dev/null; then
        # Generate additional charts using Python if available
        local chart_script="scripts/generate_charts.py"
        if [ -f "$chart_script" ]; then
            print_status "Running chart generation script..."
            python3 "$chart_script" --input "$output_dir/benchmark_results.json" --output "$output_dir/charts" || print_warning "Chart generation script failed"
        fi
    fi
    
    print_success "Additional charts generated"
}

# Function to validate results
validate_results() {
    local output_dir="$1"
    
    print_step "Validating benchmark results..."
    
    # Check if main result files exist
    local required_files=("benchmark_results.json" "benchmark_report.html" "benchmark_report.md")
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$output_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing result files: ${missing_files[*]}"
        return 1
    fi
    
    # Validate JSON format
    if command -v jq &> /dev/null; then
        if ! jq empty "$output_dir/benchmark_results.json" 2>/dev/null; then
            print_error "Invalid JSON in results file"
            return 1
        fi
    fi
    
    # Check file sizes (should not be empty)
    for file in "${required_files[@]}"; do
        if [ ! -s "$output_dir/$file" ]; then
            print_warning "Result file $file is empty"
        fi
    done
    
    print_success "Results validation passed"
}

# Function to upload results
upload_results() {
    local output_dir="$1"
    
    if [ "$UPLOAD_RESULTS" = false ]; then
        return
    fi
    
    print_step "Uploading benchmark results..."
    
    # Create archive
    local archive_name="benchmark_results_$(date +%Y%m%d_%H%M%S).tar.gz"
    local archive_path="/tmp/$archive_name"
    
    tar -czf "$archive_path" -C "$(dirname "$output_dir")" "$(basename "$output_dir")"
    
    print_status "Created archive: $archive_path"
    
    # Upload logic would go here (S3, FTP, etc.)
    # For now, just print the command that would be used
    print_status "Upload command would be: aws s3 cp $archive_path s3://benchmark-results/"
    print_warning "Upload not implemented - archive saved locally at $archive_path"
}

# Function to cleanup
cleanup() {
    print_step "Cleaning up temporary files..."
    # Add cleanup logic here if needed
    print_success "Cleanup completed"
}

# Function to print final summary
print_summary() {
    local output_dir="$1"
    local start_time="$2"
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    echo
    echo "=========================================="
    echo "🏁 GPU Agents Benchmark Complete!"
    echo "=========================================="
    echo "📊 Results Directory: $output_dir"
    echo "🔗 Latest Results: ${output_dir%/*}/latest"
    echo "⏱️  Total Duration: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)"
    echo "📈 Reports Generated:"
    
    if [ -f "$output_dir/benchmark_report.html" ]; then
        echo "   • HTML Report: $output_dir/benchmark_report.html"
    fi
    if [ -f "$output_dir/benchmark_report.md" ]; then
        echo "   • Markdown Report: $output_dir/benchmark_report.md"
    fi
    if [ -f "$output_dir/benchmark_results.json" ]; then
        echo "   • JSON Data: $output_dir/benchmark_results.json"
    fi
    if [ -f "$output_dir/benchmark_summary.csv" ]; then
        echo "   • CSV Summary: $output_dir/benchmark_summary.csv"
    fi
    
    echo
    echo "🎯 Quick Summary:"
    if [ -f "$output_dir/benchmark_results.json" ] && command -v jq &> /dev/null; then
        local max_agents=$(jq -r '.summary.max_agents_spawned // "N/A"' "$output_dir/benchmark_results.json")
        local rating=$(jq -r '.summary.overall_performance_rating // "N/A"' "$output_dir/benchmark_results.json")
        echo "   • Max Agents: $max_agents"
        echo "   • Overall Rating: $rating"
    fi
    
    echo
    echo "📖 To view results:"
    echo "   • Open $output_dir/benchmark_report.html in a web browser"
    echo "   • Quick access: ${output_dir%/*}/latest/benchmark_report.html"
    echo "   • Read $output_dir/benchmark_report.md with any markdown viewer"
    echo "   • Analyze raw data in $output_dir/benchmark_results.json"
    echo
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -o|--output)
                DEFAULT_OUTPUT_DIR="$2"
                shift 2
                ;;
            -c|--config)
                DEFAULT_CONFIG_FILE="$2"
                shift 2
                ;;
            -t|--type)
                BENCHMARK_TYPE="$2"
                if [[ ! "$BENCHMARK_TYPE" =~ ^(quick|standard|stress)$ ]]; then
                    print_error "Invalid benchmark type: $BENCHMARK_TYPE"
                    exit 1
                fi
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --no-charts)
                GENERATE_CHARTS=false
                shift
                ;;
            --upload)
                UPLOAD_RESULTS=true
                shift
                ;;
            --scalability-only)
                SCALABILITY_ONLY=true
                shift
                ;;
            --llm-only)
                LLM_ONLY=true
                shift
                ;;
            --kg-only)
                KG_ONLY=true
                shift
                ;;
            --evolution-only)
                EVOLUTION_ONLY=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set log level if verbose
    if [ "$VERBOSE" = true ]; then
        export BENCHMARK_LOG_LEVEL="debug"
    fi
    
    echo "🚀 GPU Agents Benchmark Runner"
    echo "=============================="
    
    # Show monitoring dashboard info
    print_status "Progress monitoring enabled - log file: $PROGRESS_LOG"
    if [ -f "monitor_dashboard.sh" ]; then
        print_status "Run './monitor_dashboard.sh' in another terminal for real-time monitoring"
    fi
    echo
    
    # Execute benchmark pipeline with progress logging
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔍 System Check - Validating GPU and CUDA availability" >> "$PROGRESS_LOG"
    check_requirements
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📦 Dependencies - Installing and updating benchmark dependencies" >> "$PROGRESS_LOG"
    install_dependencies
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔨 Build - Compiling benchmark binary in $([ "$BENCHMARK_TYPE" = "stress" ] && echo "release" || echo "debug") mode" >> "$PROGRESS_LOG"
    build_benchmark
    
    local output_dir
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📁 Setup - Creating timestamped output directory" >> "$PROGRESS_LOG"
    output_dir=$(setup_output_directory "$DEFAULT_OUTPUT_DIR")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 Output Directory: $output_dir" >> "$PROGRESS_LOG"
    
    # Log benchmark start with system info
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Benchmark Execution - Starting $BENCHMARK_TYPE benchmark suite" >> "$PROGRESS_LOG"
    if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎮 GPU Info: $gpu_info" >> "$PROGRESS_LOG"
    fi
    
    if ! run_benchmarks "$output_dir"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Benchmark execution failed" >> "$PROGRESS_LOG"
        print_error "Benchmark execution failed"
        exit 1
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📈 Charts - Generating additional performance visualizations" >> "$PROGRESS_LOG"
    generate_additional_charts "$output_dir"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Validation - Checking benchmark results integrity" >> "$PROGRESS_LOG"
    if ! validate_results "$output_dir"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Result validation failed" >> "$PROGRESS_LOG"
        print_error "Result validation failed"
        exit 1
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📤 Upload - Processing results upload (if enabled)" >> "$PROGRESS_LOG"
    upload_results "$output_dir"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧹 Cleanup - Finalizing benchmark execution" >> "$PROGRESS_LOG"
    cleanup
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🏁 Complete - Benchmark suite finished successfully" >> "$PROGRESS_LOG"
    print_summary "$output_dir" "$start_time"
    
    # Final progress log summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 Summary - Total execution time: ${total_duration}s ($(($total_duration / 60))m $(($total_duration % 60))s)" >> "$PROGRESS_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 Summary - Results available at: $output_dir" >> "$PROGRESS_LOG"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 Summary - Latest symlink: ${output_dir%/*}/latest" >> "$PROGRESS_LOG"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"